//! BeatBox-to-MIDI Tauri Backend
//!
//! This module provides the desktop shell for the BeatBox-to-MIDI application.
//! It handles window management and communication with the Python audio engine.

#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use serde::{Deserialize, Serialize};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use tauri::State;

/// Application state
struct AppState {
    python_process: Mutex<Option<Child>>,
}

/// Engine status response
#[derive(Serialize, Deserialize)]
struct EngineStatus {
    running: bool,
    port: u16,
}

/// Start the Python audio engine
#[tauri::command]
fn start_engine(state: State<AppState>) -> Result<EngineStatus, String> {
    let mut process_guard = state
        .python_process
        .lock()
        .map_err(|e| format!("Failed to lock state: {}", e))?;

    // Check if already running
    if process_guard.is_some() {
        return Ok(EngineStatus {
            running: true,
            port: 8765,
        });
    }

    // Start Python engine
    let python_cmd = if cfg!(target_os = "windows") {
        "python"
    } else {
        "python3"
    };

    let child = Command::new(python_cmd)
        .args(["engine/main.py"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start Python engine: {}", e))?;

    *process_guard = Some(child);

    Ok(EngineStatus {
        running: true,
        port: 8765,
    })
}

/// Stop the Python audio engine
#[tauri::command]
fn stop_engine(state: State<AppState>) -> Result<bool, String> {
    let mut process_guard = state
        .python_process
        .lock()
        .map_err(|e| format!("Failed to lock state: {}", e))?;

    if let Some(mut child) = process_guard.take() {
        child
            .kill()
            .map_err(|e| format!("Failed to kill process: {}", e))?;
        child
            .wait()
            .map_err(|e| format!("Failed to wait for process: {}", e))?;
    }

    Ok(true)
}

/// Check if engine is running
#[tauri::command]
fn is_engine_running(state: State<AppState>) -> Result<bool, String> {
    let mut process_guard = state
        .python_process
        .lock()
        .map_err(|e| format!("Failed to lock state: {}", e))?;

    if let Some(ref mut child) = *process_guard {
        // Check if process is still running
        match child.try_wait() {
            Ok(Some(_)) => {
                // Process has exited
                *process_guard = None;
                Ok(false)
            }
            Ok(None) => {
                // Still running
                Ok(true)
            }
            Err(e) => Err(format!("Failed to check process status: {}", e)),
        }
    } else {
        Ok(false)
    }
}

/// Get WebSocket URL for the engine
#[tauri::command]
fn get_engine_url() -> String {
    "ws://localhost:8765".to_string()
}

fn main() {
    tauri::Builder::default()
        .manage(AppState {
            python_process: Mutex::new(None),
        })
        .invoke_handler(tauri::generate_handler![
            start_engine,
            stop_engine,
            is_engine_running,
            get_engine_url,
        ])
        .on_window_event(|event| {
            if let tauri::WindowEvent::CloseRequested { .. } = event.event() {
                // Clean up Python process on window close
                if let Some(app_state) = event.window().try_state::<AppState>() {
                    if let Ok(mut guard) = app_state.python_process.lock() {
                        if let Some(mut child) = guard.take() {
                            let _ = child.kill();
                            let _ = child.wait();
                        }
                    }
                }
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
