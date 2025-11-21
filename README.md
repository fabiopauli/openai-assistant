# Horizon Assistant

Horizon Assistant is an advanced AI coding assistant for the terminal, powered by the OpenAI API and supporting large context windows, file and Git operations, and secure shell command execution. It is designed to act as a senior engineer, providing expert guidance, code analysis, and automation for development workflows.

---

## Overview

Horizon Assistant provides:
- Interactive CLI with rich UI (using `prompt_toolkit` and `rich`)
- Fuzzy file and snippet matching for easy file access and editing
- Deep Git integration (init, add, commit, branch, status)
- Secure shell command execution (bash, PowerShell) with user confirmation
- Large context management (128k tokens) for big projects and conversations
- Model switching between chat and reasoner modes
- Smart context truncation and file context management


## Key Features

- **OpenAI API Integration:** Uses OpenAI SDK to interact with OpenAI-compatible models (default: `gpt-5-mini` /reasoner: `gpt-5`).
- **Large Context Window:** Up to 256k tokens for conversations and file contexts.
- **Interactive CLI:** Rich UI with autocompletion, color, and tables.
- **File Operations:** Read, create, edit files and directories, with fuzzy matching and size-aware context management.
- **Git Operations:** Initialize, stage, commit, branch, and status commands, with staging checks and context awareness.
- **Shell Command Execution:** Run bash or PowerShell commands securely, with explicit user confirmation.
- **Model Switching:** Toggle between chat and reasoner models for different tasks.
- **Context Management:** Smart truncation and file context limits to avoid exceeding model limits.
- **Security:** All shell and file operations require confirmation and are restricted to the project directory.

## Setup

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```
4. Run the assistant:
   ```bash
   python main.py
   ```

## Usage

The assistant accepts commands prefixed by `/` or custom prefixes:

### Common Commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/exit` or `/quit` | Exit the assistant |
| `/clear` | Clear the screen |
| `/clear-context` | Clear conversation and file context |
| `/context` | Show context token usage stats |
| `/os` | Show OS and shell info |

### File Operations

| Command | Description |
|---------|-------------|
| `/add <path>` | Add file or directory to context (fuzzy matching supported) |
| `/folder` | Show/set/reset base directory for file operations |

### Git Operations

| Command | Description |
|---------|-------------|
| `/git init` | Initialize a Git repository |
| `/git status` | Show Git status |
| `/git add <files>` | Stage files or all (`.`) |
| `/git commit <message>` | Commit staged changes |
| `/git branch <name>` | Create and switch to a new branch |
| `/git-info` | Show detailed Git capabilities |

### Model & Reasoner

| Command | Description |
|---------|-------------|
| `/reasoner` | Toggle between chat and reasoner models |
| `/agent` | Switch to gpt-5.1 agent model |
| `/r` | One-off reasoner call |

### Security

- Shell commands (`run_bash`, `run_powershell`) require explicit confirmation.
- File and directory operations are restricted to the project directory for safety.

### Context Management

- Smart truncation keeps context within model limits.
- File contexts are limited and managed automatically.

### Fuzzy Matching

- File and snippet matching uses fuzzy logic for ease of use.

### OS Awareness

- Detects available shells and adapts commands for Linux, macOS, or Windows.

### Project Structure

- `main.py` — Main CLI and command loop
- `config.py` — Configuration, constants, and environment detection
- `utils.py` — Utility functions (fuzzy matching, context management, etc.)
- `system_prompt.txt` — System prompt for the assistant
- `requirements.txt` — Python dependencies

---

For more details, start the assistant and type `/help`.

---

© Horizon Assistant Project

### File Management

| Command | Description |
|---------|-------------|
| `/add <path>` | Add a file or directory to conversation context (supports fuzzy matching) |
| `/folder` | Show current base directory |
| `/folder <path>` | Set base directory for file operations |
| `/folder reset` | Reset base directory to current working directory |

### Git Workflow

| Command | Description |
|---------|-------------|
| `/git init` | Initialize a Git repository |
| `/git status` | Show Git status |
| `/git add <files>` | Stage specific files or all (`.`) |
| `/git commit <message>` | Commit staged changes with a message |
| `/git branch <name>` | Create and switch to a new branch |
| `/git-info` | Show detailed Git capabilities |

### Model Switching

| Command | Description |
|---------|-------------|
| `/reasoner` | Toggle between chat and reasoner models |
| `/agent` | Switch to gpt-5.1 agent model |
| `/r` | Call the reasoner model for one-off reasoning tasks |

## Security

- Running shell commands (`run_bash` or `run_powershell`) requires explicit user confirmation to prevent unauthorized or harmful commands.

## Notes

- The assistant uses fuzzy matching for file commands to ease file access and editing.
- Context size is carefully managed to avoid exceeding model limits.
- Git operations are integrated deeply, including staging and commit checks.
- The assistant aims to be a senior engineer, providing detailed explanations and thoughtful guidance.

## Project Structure

- `main.py` - Main application and CLI loop.
- `config.py` - Configuration, constants, and environment detection.
- `utils.py` - Utility functions (not detailed here).
- `system_prompt.txt` - Optional system prompt override.
- `requirements.txt` - Python dependencies.

---

For detailed usage, start the assistant and type `/help`.

---

© Horizon Assistant Project
