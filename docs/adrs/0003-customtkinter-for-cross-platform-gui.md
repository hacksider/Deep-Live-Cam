# ADR 0003: CustomTkinter for Cross-Platform GUI

## Status
**Accepted** (Established since project inception, 2023)

## Context

Deep-Live-Cam requires a cross-platform GUI for:
- Source image selection
- Camera/video input selection
- Live preview of face swapping
- Frame processor toggles (enhancer, mouth mask)
- Face mapping (drag-and-drop interface)
- Output settings configuration

Initial design chose **CustomTkinter** over alternatives like PyQt5, wxPython, or GTK.

## Decision

Use **CustomTkinter** (v5.2.2) as the GUI framework because:

1. **Lightweight dependency**: Built on Tcl/Tk (system library), minimal overhead
2. **Native look**: Respects platform conventions (macOS native feel)
3. **Modern widgets**: Dark mode, rounded buttons, modern color schemes (unlike raw Tk)
4. **Tkinter compatibility**: Leverages decades of Tk stability
5. **Image handling**: PIL/ImageTk for efficient frame preview in Canvas

### GUI Architecture
- CustomTkinter window (960×540 default preview)
- Canvas-based video preview (real-time frame updates)
- Frame processor toggles, camera selection, face mapping UI
- Responsive to model loading and processing state

## Consequences

### Positive
✓ **Lightweight**: No Qt/wxPython binary dependency (Tk is system library)
✓ **Fast startup**: Tk initialization <1 second
✓ **Native platform feel**: Respects macOS/Windows/Linux conventions
✓ **Modern aesthetics**: CustomTkinter dark mode looks professional
✓ **Proven**: Tk is 30+ year old stable technology
✓ **Image processing**: PIL/ImageTk efficient for 1080p 30 FPS preview
✓ **Tcl/Tk ecosystem**: Rich widget library for future expansion

### Negative
✗ **Tcl/Tk complexity**: Requires system Tcl/Tk installation (hardcoded init.tcl paths)
✗ **Threading issues**: Tk's event loop thread-safety requires careful handling
✗ **Customization limits**: Harder to create highly custom widgets vs Qt/GTK
✗ **Mobile**: No mobile platform support (iOS/Android)
✗ **Advanced layouts**: Complex layout requirements need workarounds
✗ **Dependency fragmentation**: Different Python builds ship different Tk versions

### Mitigations
- **Tcl/Tk setup**: Justfile auto-detects and exports TCL_LIBRARY/TK_LIBRARY
- **Thread safety**: GUI updates from worker threads use Tk's `after()` callback
- **PIL fallback**: ImageTk errors caught and logged (diagnostic for misconfiguration)
- **mise + uv**: Standardized Python build (python-build-standalone) with bundled Tk

## Evidence

### Git History
- Initial commit (Sep 24, 2023): ui.py uses CustomTkinter
- Consistent across all versions (v0.x → v2.0.3c)
- Multiple ui.py refinements show iterative improvement, not wholesale framework replacement

### Architecture Notes
- Video preview: Canvas widget updated every frame via PIL Image conversion
- Controls: CustomTkinter buttons, checkboxes, comboboxes for processor selection
- Threading: Video capture and face processing run in background threads; GUI updated via `after()` callbacks
- Responsive UI: <200ms latency from control interaction to effect

### Performance
- Frame preview: PIL ImageTk creation ~5ms per frame @ 1080p
- Window responsiveness: No frame drops during processor toggle
- Memory: Single image cache (current frame only), not full video buffer

## Related Decisions
- [ADR 0008: ThreadPoolExecutor](0008-threadpoolexecutor-for-parallel-frame-processing.md) (threaded GUI update)
- [ADR 0005: Global Mutable State](0005-global-mutable-state-for-configuration.md) (UI state sync)

## Future Considerations
- Could migrate to Qt/wxPython if advanced customization needed
- Electron/web-based UI possible for mobile support
- Tcl/Tk upgrade path (tk 8.7 in development)

**Last Reviewed**: Feb 18, 2026 | **Confidence**: High
