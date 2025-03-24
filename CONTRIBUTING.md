# Collaboration Guidelines and Codebase Quality Standards

To ensure smooth collaboration and maintain the high quality of our codebase, please adhere to the following guidelines:

## Branching Strategy

*   **`premain`**:
    *   Always push your changes to the `premain` branch initially.
    *   This safeguards the `main` branch from unintentional disruptions.
    *   All tests will be performed on the `premain` branch.
    *   Changes will only be merged into `main` after several hours or days of rigorous testing.
*   **`experimental`**:
    *   For large or potentially disruptive changes, use the `experimental` branch.
    *   This allows for thorough discussion and review before considering a merge into `main`.

## Pre-Pull Request Checklist

Before creating a Pull Request (PR), ensure you have completed the following tests:

### Functionality

*   **Realtime Faceswap**:
    *   Test with face enhancer **enabled** and **disabled**.
*   **Map Faces**:
    *   Test with both options (**enabled** and **disabled**).
*   **Camera Listing**:
    *   Verify that all cameras are listed accurately.

### Stability

*   **Realtime FPS**:
    *   Confirm that there is no drop in real-time frames per second (FPS).
*   **Boot Time**:
    *   Changes should not negatively impact the boot time of either the application or the real-time faceswap feature.
*   **GPU Overloading**:
    *   Test for a minimum of 15 minutes to guarantee no GPU overloading, which could lead to crashes.
*   **App Performance**:
    *   The application should remain responsive and not exhibit any lag.
