//! [`InspectorPlugin`] — connect a [`crate::Harness`] to a `kittest_inspector` process for
//! live debugging.
//!
//! The plugin spawns the inspector as a child process with piped stdin/stdout. After every
//! step it sends a rendered frame + accesskit tree + source-location info, then blocks on
//! the inspector's reply. If the inspector returns events (Control mode), the plugin queues
//! them and calls [`crate::Harness::advance_frame`] twice — once to apply the event and once
//! so its visible effect shows on the next frame sent to the inspector.
//!
//! Auto-registered on harness creation when the [`INSPECTOR_ENV_VAR`] env var is truthy.

use std::any::Any;
use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Write as _};
use std::panic::Location;
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{LazyLock, OnceLock};

use egui::accesskit;
use egui::mutex::Mutex;

use crate::inspector_api::{
    Frame, HarnessMessage, InspectorReply, SourceView, read_message, write_message,
};
use crate::{Harness, Plugin, TestResult};

/// Environment variable: when set to a truthy value, every harness auto-launches an inspector.
pub const INSPECTOR_ENV_VAR: &str = "KITTEST_INSPECTOR";

/// Environment variable: explicit path to the `kittest_inspector` binary.
pub const INSPECTOR_PATH_ENV_VAR: &str = "KITTEST_INSPECTOR_PATH";

/// Errors that can occur attaching or talking to the inspector.
#[derive(Debug)]
pub enum InspectorError {
    /// Failed to launch the `kittest_inspector` binary.
    Launch(std::io::Error),
    /// Failed to set up the child's stdio pipes.
    Pipe(String),
}

impl std::fmt::Display for InspectorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Launch(err) => write!(
                f,
                "failed to launch kittest_inspector (set {INSPECTOR_PATH_ENV_VAR} or put it on PATH): {err}"
            ),
            Self::Pipe(msg) => write!(f, "inspector pipe setup failed: {msg}"),
        }
    }
}

impl std::error::Error for InspectorError {}

/// Plugin that streams frames to an external `kittest_inspector` binary.
///
/// Typical use is to let [`Harness::from_builder`] auto-register this plugin based on the
/// [`INSPECTOR_ENV_VAR`] environment variable. For manual wiring, construct one with
/// [`Self::launch`] and pass to [`crate::HarnessBuilder::with_plugin`].
pub struct InspectorPlugin {
    conn: Connection,
}

impl InspectorPlugin {
    /// Launch a `kittest_inspector` child process and attach this plugin to it.
    ///
    /// # Errors
    /// If the inspector binary cannot be launched or its stdio pipes fail to set up.
    pub fn launch(label: Option<String>) -> Result<Self, InspectorError> {
        Ok(Self {
            conn: Connection::launch(label)?,
        })
    }
}

impl<S> Plugin<S> for InspectorPlugin {
    fn after_step(&mut self, harness: &mut Harness<'_, S>) {
        self.conn.drive(harness);
    }

    fn on_test_result(&mut self, _harness: &mut Harness<'_, S>, _result: TestResult<'_>) {
        self.conn.say_goodbye();
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// The inspector's child-process connection + step counter. Private — [`InspectorPlugin`] is
/// the public wrapper.
struct Connection {
    writer: BufWriter<ChildStdin>,
    reader: BufReader<ChildStdout>,
    _child: Child,
    step: u64,
    label: Option<String>,
    broken: bool,
}

impl Connection {
    fn launch(label: Option<String>) -> Result<Self, InspectorError> {
        let bin = std::env::var(INSPECTOR_PATH_ENV_VAR)
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("kittest_inspector"));

        // Important: do NOT inherit stderr. The cargo-test / nextest stderr capture pipe can
        // close between tests while the inspector is still alive; a later `eprintln!` in the
        // inspector would then panic ("failed printing to stderr: Broken pipe") and take the
        // window down. The inspector keeps its own log file for diagnostics.
        let mut child = Command::new(&bin)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(InspectorError::Launch)?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| InspectorError::Pipe("missing child stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| InspectorError::Pipe("missing child stdout".into()))?;

        Ok(Self {
            writer: BufWriter::new(stdin),
            reader: BufReader::new(stdout),
            _child: child,
            step: 0,
            label,
            broken: false,
        })
    }

    /// Block at the inspector until it replies with no events, re-rendering each time the
    /// user injects an event via Control mode. Mirrors the outer `step()` loop's behavior
    /// using `Harness::advance_frame` (no plugin dispatch) to avoid recursing.
    fn drive<S>(&mut self, harness: &mut Harness<'_, S>) {
        loop {
            let image = match harness.render() {
                Ok(img) => img,
                Err(err) => {
                    #[expect(clippy::print_stderr)]
                    {
                        eprintln!("egui_kittest inspector: render failed: {err}");
                    }
                    return;
                }
            };
            let tree = harness.accesskit_tree_update().cloned();
            let ppp = harness.ctx.pixels_per_point();
            let call_site = harness.entry_location();
            let event_sites = harness.consumed_event_locations().to_vec();
            let events = self.send_step(&image, ppp, tree, call_site, &event_sites);
            if events.is_empty() {
                return;
            }
            for event in events {
                harness.input_mut().events.push(event);
            }
            // Inspector-driven events don't carry a test-source location.
            // Advance once to apply, then again so the effect is visible on the next frame
            // we send back to the inspector.
            harness.advance_frame();
            harness.advance_frame();
        }
    }

    fn send_step(
        &mut self,
        image: &image::RgbaImage,
        pixels_per_point: f32,
        accesskit: Option<accesskit::TreeUpdate>,
        call_site: Option<&'static Location<'static>>,
        event_sites: &[&'static Location<'static>],
    ) -> Vec<egui::Event> {
        if self.broken {
            return Vec::new();
        }
        self.step = self.step.saturating_add(1);
        let frame = Frame {
            step: self.step,
            width: image.width(),
            height: image.height(),
            pixels_per_point,
            rgba: image.as_raw().clone(),
            accesskit,
            label: self.label.clone(),
            source: build_source_view(call_site, event_sites),
        };
        if let Err(err) = write_message(&mut self.writer, &HarnessMessage::Frame(Box::new(frame))) {
            #[expect(clippy::print_stderr)]
            {
                eprintln!("egui_kittest inspector: send failed: {err}");
            }
            self.broken = true;
            return Vec::new();
        }
        match read_message::<_, InspectorReply>(&mut self.reader) {
            Ok(InspectorReply::Continue { events }) => events,
            Err(err) => {
                #[expect(clippy::print_stderr)]
                {
                    eprintln!("egui_kittest inspector: read failed: {err}");
                }
                self.broken = true;
                Vec::new()
            }
        }
    }

    fn say_goodbye(&mut self) {
        if self.broken {
            return;
        }
        let _ = write_message(&mut self.writer, &HarnessMessage::Goodbye);
        let _ = self.writer.flush();
    }
}

impl Drop for Connection {
    fn drop(&mut self) {
        self.say_goodbye();
    }
}

/// Build the [`SourceView`] payload for a frame: pick the `.run()`/`.step()` caller's file
/// as the anchor, and record each event's line within that same file.
///
/// `#[track_caller]` chains through the entire event-queuing API, so each `Location` points
/// directly at the user's test source — no backtrace walking needed.
fn build_source_view(
    call_site: Option<&'static Location<'static>>,
    event_sites: &[&'static Location<'static>],
) -> Option<SourceView> {
    let call = call_site?;
    let path = call.file().to_owned();
    let event_lines = event_sites
        .iter()
        .filter(|loc| loc.file() == path)
        .map(|loc| loc.line())
        .collect();
    Some(SourceView {
        path: path.clone(),
        contents: read_source_file(&path),
        call_site_line: Some(call.line()),
        event_lines,
    })
}

/// Read the full contents of a source file, cached per path (including negative results).
fn read_source_file(path: &str) -> Option<String> {
    static CACHE: LazyLock<Mutex<HashMap<String, Option<String>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));
    let mut cache = CACHE.lock();
    cache
        .entry(path.to_owned())
        .or_insert_with(|| std::fs::read_to_string(path).ok())
        .clone()
}

/// Read [`INSPECTOR_ENV_VAR`] once and cache. Exposed to [`crate::Harness::from_builder`]
/// so it can auto-register an [`InspectorPlugin`].
pub(crate) fn env_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var(INSPECTOR_ENV_VAR) {
        Ok(value) => matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        ),
        Err(_) => false,
    })
}
