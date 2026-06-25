# ComputerVision — Eye-Tracking Learning Project

## Project goal

Build a webcam eye-tracking app that estimates **where on the screen the user is looking** (gaze point), with the best precision achievable given real-world constraints (lighting, webcam quality, glasses). Possible far-future use: a horror game. Not a concern now.

## Who I'm working with

The user is **learning computer vision through this project** and starts with **zero CV knowledge**. This file is the contract for HOW to help, not just what to build.

## Teaching contract — READ EVERY TIME

1. **Do NOT give straightforward/complete answers unless the user explicitly asks for them.** Default to hints, questions, partial direction, and "try this and see." Let the user do the thinking and the typing. When in doubt, withhold the solution and nudge instead.
2. **Start every new step with a concise, high-level, step-by-step plan.** Explain WHAT should happen at a high level. Give deeper detail / code / specifics ONLY when asked.
3. **Detail taper:** more hints and hand-holding at the start of the project; progressively fewer as the project advances and the user's skill grows. Calibrate to demonstrated understanding.
4. **Offer extra ideas/suggestions freely** whenever a useful one appears (alternative approaches, tools, pitfalls, optimizations) — clearly marked as optional asides.
5. When the user is stuck after genuine effort, escalate help gradually: hint → stronger hint → partial solution → full answer. Don't jump straight to the answer.

## Environment / permissions

- Full bash control **except**: `sudo`, `rm`, `git push` (ask for their use).
- Platform: macOS (darwin). Webcam-based.
- Not a git repo yet.

## Working style

- Caveman comms mode may be active (terse). Teaching content and plans still complete — compress style, never substance.
- Prefer letting the user run/observe things over me asserting outcomes.
- **Maintain `instructions.txt`** — the running record of Python version + every lib in use (for future port to newer Python). Update it whenever a dependency is added/removed.

## Status

Step 1 (env) DONE. Python 3.12.13 venv at `.venv`. Installed + verified: opencv-python 4.13, numpy 2.5, mediapipe 0.10.35. mediapipe legacy `solutions` API gone → use Tasks API (`FaceLandmarker`, needs a `.task` model download). Next: webcam capture loop → live window (step 1 finish), then face/iris landmarks.
