# Aegis Nexus 9.0 - Showcase

## Overview
This repository contains key components of the **Aegis Nexus 9.0** system, a sovereign intelligence architecture designed for high-assurance governance and resilience.

## Directory Structure

### `/core`
Contains the heart of the system logic.
- **Governance**: Z3-based formal verification modules.
- **Logic**: Core Rust/Python bridges for high-performance decision making.

### `/reality_anchor`
Implements the unique "Reality Anchor" consensus mechanism.
- **Sensors**: Sensor fusion logic.
- **Fusion Engine**: Data aggregation and truth verification.

### `/api`
The backend API layer built with FastAPI.
- **Middleware**: Custom security and SBAC (Sovereign Based Access Control) middleware.
- **Endpoints**: System interaction points.

### `/frontend`
React-based dashboard source code.
- **Src**: UI components and state management.

### `/infrastructure`
Deployment and configuration artifacts.
- **Terraform**: Infrastructure as Code.
- **Docker**: Container definitions.

## Getting Started
This is a partial showcase of the larger Aegis Nexus system.
To explore the core logic, examine `core/governance/z3_verifier.py`.
