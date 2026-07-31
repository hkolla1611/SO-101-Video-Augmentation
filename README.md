# SO-101 Video Augmentation

Physical AI video augmentation pipeline for SO-101 robotic demonstrations using Hugging Face LeRobot, NVIDIA Physical AI Data Factory (PAIDF), Cosmos Transfer 2.5, NVIDIA Jetson Orin Nano, and Vast.ai GPU infrastructure.

This project demonstrates a workflow for transforming real SO-101 robot demonstration videos into visually diverse synthetic variants while preserving robot motion, object interaction, camera viewpoint, scene geometry, and temporal consistency.

---

## Project Overview

Robot-learning systems depend on diverse training demonstrations. Collecting the same robotic task repeatedly under different lighting conditions, backgrounds, environments, and visual appearances requires considerable physical setup and data-collection time.

This project explores generative video augmentation as an additional source of visual diversity.

```text
SO-101 Robot Demonstration
          ↓
Hugging Face LeRobot
          ↓
Camera Observation
          ↓
Video Preparation
          ↓
Vast.ai GPU
          ↓
NVIDIA Physical AI Augmentation
          ↓
Generative Video Processing
          ↓
Augmented Robot Video
```

The objective is to modify visual properties of a demonstration while preserving important task information such as:

- Robot motion
- Gripper behavior
- Object motion
- Camera viewpoint
- Scene geometry
- Temporal consistency

The resulting synthetic videos can be explored for downstream robot-learning and imitation-learning experiments.

---

## System Architecture

```text
             Human Operator
                    │
                    ▼
             SO-101 Leader
                    │
              Teleoperation
                    ▼
             SO-101 Follower
                    │
                    ▼
          Object Manipulation
                    │
                    ▼
              Camera Capture
                    │
                    ▼
        NVIDIA Jetson Orin Nano
                    │
              Hugging Face
                 LeRobot
                    │
                    ▼
          Robot Demonstration
                    │
                    ▼
          Video Preparation
          FFmpeg / FFprobe
                    │
                    ▼
              Vast.ai GPU
                    │
                    ▼
        NVIDIA Physical AI
        Augmentation Pipeline
                    │
                    ▼
           Generative Model
                    │
                    ▼
            Augmented MP4
                    │
                    ▼
          Output Validation
```

---

## Hardware

### SO-101 Robot System

The physical robotics setup consists of:

- SO-101 Leader Arm
- SO-101 Follower Arm
- Feetech servo motors
- Camera for visual observations
- Leader-follower teleoperation

During demonstration collection:

```text
Human Operator
      ↓
SO-101 Leader
      ↓
SO-101 Follower
      ↓
Object Manipulation
      ↓
Camera Observation
```

The leader arm is manually controlled while the follower reproduces the demonstrated motion.

### NVIDIA Jetson Orin Nano

The NVIDIA Jetson Orin Nano is used as the edge computing platform for robot operation and data collection.

The Jetson environment handles:

- SO-101 communication
- Robot calibration
- Leader-follower teleoperation
- Camera access
- Robot state collection
- Robot action collection
- LeRobot dataset recording
- Video capture

### Cloud GPU

GPU-intensive generative processing is performed using Vast.ai.

This separates the workload into:

```text
Jetson Orin Nano
      │
      ├── Robot control
      ├── Teleoperation
      └── Data collection
              │
              ▼
       Robot Demonstration
              │
              ▼
           Vast.ai
              │
      ├── GPU inference
      ├── Video generation
      └── Video augmentation
```

The Jetson handles the physical robot while cloud GPUs handle computationally expensive generative processing.

---

## Software Stack

| Component | Purpose |
|---|---|
| SO-101 | Robotic manipulation platform |
| Hugging Face LeRobot | Robot control and dataset handling |
| NVIDIA Jetson Orin Nano | Edge robot control and data collection |
| NVIDIA PAIDF | Physical AI augmentation environment |
| Cosmos Transfer 2.5 | Generative video transformation |
| Vast.ai | Cloud GPU infrastructure |
| NVIDIA NGC | NVIDIA container registry |
| Hugging Face Hub | Dataset and model access |
| FFmpeg | Video preprocessing and conversion |
| FFprobe | Video inspection and validation |
| OpenCV | Computer vision and video processing |
| Python | Pipeline execution and utilities |

---

## SO-101 Demonstration Data

Robot demonstrations are collected using Hugging Face LeRobot with the SO-101 leader-follower setup.

```text
Operator moves Leader
        ↓
Leader joint positions
        ↓
Follower reproduces motion
        ↓
Robot interacts with object
        ↓
Camera captures demonstration
        ↓
LeRobot stores episode
```

A LeRobot episode can contain:

```text
Robot Actions
Robot States
Camera Observations
Episode Information
Task Metadata
```

Existing SO-101 LeRobot datasets can also be used as video sources for augmentation experiments.

---

## Input Video

The camera observation from a robot demonstration is used as the visual input to the augmentation pipeline.

Example:

```text
input.mp4
```

Before augmentation, the video can be inspected using FFprobe:

```bash
ffprobe -v error \
  -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames \
  -show_entries format=duration,size \
  -of default=noprint_wrappers=1 \
  input.mp4
```

This verifies:

- Codec
- Resolution
- Frame rate
- Frame count
- Duration
- File size

---

## Video Preparation

Robot videos may require preprocessing before generative augmentation.

Preparation can include:

- Codec conversion
- Resolution adjustment
- FPS conversion
- Clip trimming
- Frame-count adjustment
- Video validation

The preprocessing workflow is:

```text
Raw Robot Video
       ↓
     Inspect
       ↓
Prepare / Convert
       ↓
Model-Compatible Video
```

FFmpeg and FFprobe are used to prepare and validate the video before GPU inference.

---

## NVIDIA Physical AI Video Augmentation

The prepared robot demonstration is processed using NVIDIA Physical AI video augmentation tooling on GPU infrastructure.

```text
Robot Video
     ↓
Augmentation Configuration
     ↓
GPU Inference
     ↓
Generative Video Processing
     ↓
Synthetic Video Variant
```

The goal is to change visual characteristics of the scene while preserving the underlying robot manipulation.

Potential visual transformations include:

- Lighting conditions
- Environmental appearance
- Background appearance
- Scene conditions
- Other model-supported visual variations

---

## Augmented Video Output

The augmentation pipeline produces a new MP4 containing a synthetic visual variant of the original robot demonstration.

The generated video can be validated with:

```bash
ffprobe -v error \
  -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames \
  -show_entries format=duration,size \
  -of default=noprint_wrappers=1 \
  output.mp4
```

A generated output from the augmentation workflow was validated as:

```text
codec_name=h264
width=640
height=480
avg_frame_rate=30/1
nb_frames=739
duration=24.633008
size=24682693
```

This confirms successful generation of a valid H.264 MP4 output.

---

## Original vs Augmented Video

```text
Original SO-101 Demonstration
              │
              ▼
      Generative Augmentation
              │
              ▼
Augmented SO-101 Demonstration
```

The visual environment can change while the underlying manipulation remains recognizable.

Important properties to preserve include:

- Robot geometry
- Robot trajectory
- Gripper motion
- Manipulated object
- Object trajectory
- Camera viewpoint
- Scene structure
- Temporal consistency

---

# NVIDIA PAIDF

NVIDIA Physical AI Data Factory (PAIDF) provides the environment used to explore Physical AI video augmentation workflows.

The PAIDF container used for the environment is:

```text
nvcr.io/nvidia/paidf-augmentation:1.0.0
```

The container was configured as a private Vast.ai template using NVIDIA NGC authentication.

---

## Important PAIDF Paths

Main application:

```text
/app
```

PAIDF Python environment:

```text
/app/.venv
```

PAIDF CLI:

```text
/app/modules/cli.py
```

Configuration directory:

```text
/app/configs
```

Cosmos Transfer 2.5:

```text
/opt/cosmos-transfer2.5
```

---

## Environment Validation

Check the GPU:

```bash
nvidia-smi
```

Navigate to PAIDF:

```bash
cd /app
```

Check the Python environment:

```bash
/app/.venv/bin/python --version
```

Set the Python module path:

```bash
export PYTHONPATH=/app:/app/modules
```

Check the PAIDF CLI:

```bash
/app/.venv/bin/python /app/modules/cli.py --help
```

Navigate to Cosmos Transfer:

```bash
cd /opt/cosmos-transfer2.5
```

These commands provide the basic checks required after launching the PAIDF environment on Vast.ai.

---

# Cosmos Transfer 2.5

Cosmos Transfer 2.5 provides generative video transformation capabilities with structural control.

The environment supports control modalities including:

```text
Edge
Depth
Segmentation
Visual / Blur
Multi-Control
```

These controls provide information from the original video to guide the generated result.

This is particularly important for robotics because visual augmentation should avoid unnecessarily modifying the physical structure and motion of the demonstrated task.

---

## Edge Control

Edge control uses structural information from the source video to guide generation.

```text
         Original Robot Video
                  │
           ┌──────┴──────┐
           │             │
           ▼             ▼
     Edge Structure    Prompt
           │             │
           └──────┬──────┘
                  │
                  ▼
         Cosmos Transfer 2.5
                  │
                  ▼
           Generated Video
```

This can help preserve structural features of the robot, objects, and scene while changing visual appearance.

---

## Distilled Edge Configuration

Cosmos Transfer 2.5 provides a distilled Edge configuration intended for faster short-video generation.

```text
Model: edge/distilled
Sampling Steps: 4
```

Example inference specification:

```json
{
  "name": "so101_edge_distilled",
  "prompt_path": "/app/data/so101/prompt.txt",
  "video_path": "/app/data/so101/input.mp4",
  "guidance": 3,
  "num_steps": 4,
  "edge": {
    "control_weight": 1.0
  }
}
```

---

## Example Augmentation Prompt

```text
A realistic indoor robotics scene showing an SO-101 robotic arm
performing a manipulation task. Preserve the robot motion,
object positions, camera viewpoint, and scene geometry while
changing the environmental appearance and lighting conditions.
```

The prompt describes the desired visual transformation while explicitly encouraging preservation of the manipulation trajectory and physical scene structure.

---

## Cosmos Inference

Navigate to Cosmos Transfer:

```bash
cd /opt/cosmos-transfer2.5
```

Example distilled Edge inference:

```bash
/app/.venv/bin/python examples/inference.py \
  -i /app/data/so101/edge_spec.json \
  -o /app/outputs/so101 \
  --model=edge/distilled
```

---

## Model Checkpoints

Cosmos checkpoints should be stored outside the GitHub repository.

A Hugging Face cache location can be configured with:

```bash
mkdir -p /workspace/hf-cache

export HF_HOME=/workspace/hf-cache
```

Authenticate with Hugging Face when required:

```bash
hf auth login
```

Model access depends on the requirements and licenses of the selected NVIDIA models.

Model checkpoints and caches should not be committed to this repository.

---

# Why Video Augmentation?

Collecting physical robot demonstrations requires:

```text
Robot Hardware
      +
Human Operator
      +
Physical Environment
      +
Object Setup
      +
Camera Recording
      +
Execution Time
```

Creating visual diversity through physical data collection may require repeatedly recreating the same task:

```text
Task + Environment A
Task + Environment B
Task + Environment C
Task + Environment D
```

Generative augmentation provides another approach:

```text
One Physical Demonstration
            │
            ▼
     Generative Model
            │
     ┌──────┼──────┐
     ▼      ▼      ▼
 Variant  Variant  Variant
    A        B        C
```

This does not replace real-world demonstrations. Instead, it provides a way to investigate whether existing demonstrations can be supplemented with additional visual diversity.

---

# Why Video Instead of Frame Augmentation?

Traditional computer vision augmentation can transform individual images using operations such as:

```text
Crop
Flip
Brightness
Contrast
Noise
```

A robot demonstration is different because it represents a continuous trajectory.

```text
Reach
  ↓
Approach
  ↓
Grasp
  ↓
Lift
  ↓
Move
  ↓
Place
```

Transforming individual frames independently could introduce inconsistencies between frames.

Generative video augmentation instead attempts to preserve coherent motion throughout the sequence.

For Physical AI, this means maintaining:

```text
Visual Diversity
       +
Temporal Consistency
       +
Physical Structure
       +
Task Semantics
```

---

# Robot-Learning Integration

The longer-term goal is to explore augmented observations as additional visual data for robot-learning experiments.

```text
              Real Demonstrations
                     │
             ┌───────┴───────┐
             │               │
             ▼               ▼
       Original Data    Video Augmentation
                             │
                             ▼
                      Synthetic Variants
             │               │
             └───────┬───────┘
                     │
                     ▼
               Training Dataset
                     │
                     ▼
              Imitation Learning
                     │
                     ▼
                    Policy
                     │
                     ▼
                SO-101 Robot
```

This allows future experiments comparing:

```text
Real Demonstrations Only
```

against:

```text
Real Demonstrations
        +
Augmented Visual Observations
```

The comparison can help determine whether synthetic visual diversity improves policy robustness and generalization.

---

# Output Validation

Generated robot videos should be evaluated technically and visually.

## Technical Validation

Use FFprobe:

```bash
ffprobe -v error \
  -show_entries stream=codec_name,width,height,avg_frame_rate,nb_frames \
  -show_entries format=duration,size \
  -of default=noprint_wrappers=1 \
  output.mp4
```

Check:

- Codec
- Resolution
- FPS
- Frame count
- Duration
- File size

## Visual Validation

Inspect the generated video for:

- Robot arm geometry
- Gripper state
- Manipulated object consistency
- Object position
- Robot trajectory
- Motion direction
- Camera viewpoint
- Scene geometry
- Temporal artifacts

A technically valid video is not necessarily a valid robot-learning demonstration. The manipulation behavior must also remain consistent.

---

# Security

Authentication credentials must never be committed to GitHub.

Do **not** upload:

```text
NVIDIA NGC API Keys
Vast.ai API Keys
Hugging Face Tokens
SSH Private Keys
Cloud Credentials
```

Recommended `.gitignore`:

```gitignore
# Secrets
.env
*.key
*.pem

# Python
.venv/
__pycache__/
*.pyc

# Model checkpoints
hf-cache/
checkpoints/
models/

# Large datasets
datasets/
data/raw/

# Generated outputs
outputs/*
!outputs/.gitkeep
```

---

# Future Work

Future extensions include:

- Generate multiple visual variants of SO-101 demonstrations
- Run additional PAIDF and Cosmos Transfer 2.5 experiments
- Experiment with Edge, Depth, Segmentation, and multi-control conditioning
- Automate batch video augmentation
- Build original-vs-augmented comparisons
- Convert synthetic observations into LeRobot-compatible datasets
- Train imitation-learning policies using real demonstrations
- Train policies using real and augmented observations
- Compare policy performance with and without augmentation
- Evaluate policy robustness under unseen visual environments

---

# Technologies

### Physical AI & Robotics

SO-101, NVIDIA Jetson Orin Nano, Hugging Face LeRobot, Feetech Servos, Leader-Follower Teleoperation, Robot Demonstration Collection

### Generative AI

NVIDIA Physical AI Data Factory (PAIDF), Cosmos Transfer 2.5, Generative Video Augmentation, Synthetic Robot Data

### Infrastructure

Vast.ai, NVIDIA NGC, Hugging Face Hub, Linux, GPU Computing

### Video Processing

FFmpeg, FFprobe, OpenCV, H.264

### Robot Learning

Imitation Learning, Behavioral Cloning, Demonstration Collection, Synthetic Data Augmentation

---

# Disclaimer

This repository documents an experimental Physical AI and robotics workflow.

NVIDIA PAIDF, Cosmos, Hugging Face LeRobot, associated datasets, containers, and models are subject to their respective licenses and usage requirements.

Model checkpoints, third-party datasets, authentication credentials, and proprietary assets are not redistributed through this repository.
