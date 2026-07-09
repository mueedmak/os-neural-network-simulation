# Multi-Core Neural Network Simulation

A Linux-based simulation of a feedforward neural network that models each network layer as a separate process and each neuron as an individual thread.

## Features

- Process-based neural network architecture
- Thread-per-neuron execution
- Parallel forward propagation
- Backward signal propagation
- Inter-process communication using pipes
- Thread synchronization using mutexes/semaphores
- Dynamic layer and neuron configuration
- Input/output through text files

## Technologies

- C++
- POSIX Threads (pthread)
- Linux
- Pipes (IPC)
- Mutexes
- Semaphores

## Project Structure

os.cpp         Main implementation
input.txt      Sample network inputs
output.txt     Generated output
Report.pdf     Project documentation

## How to Build

```bash
g++ os.cpp -pthread -o os

## Run


```bash
./os


