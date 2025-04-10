"""
Interactive Mode for PyEvo Optimization

This module provides interactive control functionality for evolutionary optimization,
allowing users to:
1. Pause/resume optimization
2. Adjust parameters during runtime 
3. View real-time statistics
4. Save/load checkpoints
"""

import time
import threading
import sys
import os
import json
from pathlib import Path
import numpy as np

from pyevo.utils.acceleration import save_checkpoint, load_checkpoint

class InteractiveOptimizer:
    """
    Interactive wrapper for PyEvo optimizers that provides runtime control.
    
    Features:
    - Pause/resume optimization
    - Adjust parameters during runtime
    - Real-time statistics
    - Checkpointing
    - Visualization support
    """
    
    def __init__(self, optimizer, fitness_function, max_iterations=None, checkpoint_dir=None):
        """
        Initialize the interactive optimizer.
        
        Args:
            optimizer: Any PyEvo optimizer instance (SNES, CMA-ES, PSO)
            fitness_function: Function to evaluate solutions
            max_iterations: Maximum number of iterations (default: None = unlimited)
            checkpoint_dir: Directory for automatic checkpoints (default: None = no auto checkpoints)
        """
        self.optimizer = optimizer
        self.fitness_function = fitness_function
        self.max_iterations = max_iterations
        self.checkpoint_dir = checkpoint_dir
        
        # Add visualization callback
        self.visualization_callback = None
        
        # Create checkpoint directory if it doesn't exist
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            
        # Control flags
        self.paused = False
        self.running = False
        self.stop_requested = False
        
        # Statistics
        self.current_iteration = 0
        self.best_fitness = float('-inf')
        self.start_time = None
        self.elapsed_time = 0
        self.iter_history = []
        self.fitness_history = []
        self.sigma_history = []
        
        # For parameter adjustment
        self.alpha_adjustment = 1.0  # Multiplier for learning rate
        
        # Command processing thread
        self.command_thread = None
        
    def start(self):
        """
        Start the optimization process with interactive control.
        
        This runs the main optimization loop while checking for pause/stop commands.
        
        Returns:
            tuple: (best_solution, best_fitness) after optimization completes
        """
        if self.running:
            print("Optimization is already running.")
            return
            
        self.running = True
        self.stop_requested = False
        self.start_time = time.time()
        
        # Start command processing thread
        self.command_thread = threading.Thread(target=self._process_commands)
        self.command_thread.daemon = True
        self.command_thread.start()
        
        print("Interactive optimization started.")
        print("Commands: pause, resume, stop, stats, save, params, help")
        
        try:
            # Main optimization loop
            while not self.stop_requested:
                if self.max_iterations and self.current_iteration >= self.max_iterations:
                    print("Maximum iterations reached.")
                    break
                    
                # Check if paused
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # Run one iteration
                improvement = self._run_iteration()
                
                # Auto-checkpoint if enabled
                if self.checkpoint_dir and self.current_iteration % 10 == 0:
                    checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{self.current_iteration}.npz")
                    session_info = self._get_session_info()
                    save_checkpoint(self.optimizer, session_info, checkpoint_path)
                
                # Early stopping check
                if improvement < 1e-6:
                    print("Early stopping: no significant improvement")
                    break
            
            print("Optimization complete.")
            self.running = False
            self._print_stats()
            
        except KeyboardInterrupt:
            print("Optimization interrupted by user.")
            self.running = False
        
        return self.optimizer.get_best_solution(), self.best_fitness
    
    def _get_session_info(self):
        """Get the current session information for checkpointing."""
        return {
            "current_iteration": self.current_iteration,
            "best_fitness": float(self.best_fitness),
            "elapsed_time": time.time() - self.start_time,
            "iter_history": self.iter_history,
            "fitness_history": [float(f) for f in self.fitness_history],
            "sigma_history": [float(s) for s in self.sigma_history],
            "alpha_adjustment": float(self.alpha_adjustment)
        }
    
    def _run_iteration(self):
        """Run a single iteration of the optimization."""
        # Generate solutions
        solutions = self.optimizer.ask()
        
        # Evaluate fitness
        fitnesses = [self.fitness_function(x) for x in solutions]
        
        # Update optimizer with adjusted learning rate
        improvement = self.optimizer.tell(fitnesses)
        
        # Update statistics
        self.current_iteration += 1
        self.best_fitness = max(self.best_fitness, max(fitnesses))
        self.iter_history.append(self.current_iteration)
        self.fitness_history.append(self.best_fitness)
        
        # Get optimizer stats - handle different optimizer types
        stats = self.optimizer.get_stats()
        if "sigma_mean" in stats:
            self.sigma_history.append(float(stats["sigma_mean"]))
        elif "sigma" in stats:
            self.sigma_history.append(float(stats["sigma"]))
        else:
            # For optimizers without sigma (e.g. PSO)
            self.sigma_history.append(0.0)
        
        # Call visualization callback if provided
        if self.visualization_callback:
            self.visualization_callback(self.optimizer, self.current_iteration)
            
        # Print progress every 10 iterations
        if self.current_iteration % 10 == 0:
            self._print_progress()
            
        return improvement
    
    def _process_commands(self):
        """Process user commands in a separate thread."""
        while self.running:
            try:
                command = input()
                self._handle_command(command)
            except EOFError:
                # Handle EOFError when input() is interrupted
                pass
    
    def _handle_command(self, command):
        """Handle user commands."""
        command = command.lower().strip()
        
        if command == "pause":
            self.paused = True
            print("Optimization paused. Type 'resume' to continue.")
            
        elif command == "resume":
            self.paused = False
            print("Optimization resumed.")
            
        elif command == "stop":
            self.stop_requested = True
            print("Stopping optimization...")
            
        elif command == "stats":
            self._print_stats()
            
        elif command == "save":
            self._save_checkpoint()
            
        elif command.startswith("alpha "):
            try:
                # Adjust learning rate by parsing the command
                value = float(command.split()[1])
                self.alpha_adjustment = value
                print(f"Learning rate multiplier set to {value}")
            except (IndexError, ValueError):
                print("Invalid alpha value. Usage: alpha <value>")
                
        elif command.startswith("params"):
            self._print_params()
            
        elif command == "help":
            self._print_help()
        
        else:
            print("Unknown command. Type 'help' for available commands.")
    
    def _print_progress(self):
        """Print progress information."""
        elapsed = time.time() - self.start_time
        print(f"Iteration {self.current_iteration}: Best fitness = {self.best_fitness:.6f} [{elapsed:.1f}s]")
    
    def _print_stats(self):
        """Print detailed statistics about the optimization."""
        print("\n--- Optimization Statistics ---")
        print(f"Iterations: {self.current_iteration}")
        print(f"Best fitness: {self.best_fitness:.6f}")
        
        elapsed = time.time() - self.start_time
        print(f"Elapsed time: {elapsed:.2f}s ({self.current_iteration / elapsed:.2f} iter/s)")
        
        # Get optimizer-specific stats
        stats = self.optimizer.get_stats()
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        print("----------------------------")
    
    def _print_params(self):
        """Print current optimizer parameters."""
        print("\n--- Optimizer Parameters ---")
        for key, value in self.optimizer.__dict__.items():
            if not key.startswith('_') and not isinstance(value, (np.ndarray, list)) and key != 'rng':
                print(f"{key}: {value}")
                
        print(f"alpha_adjustment: {self.alpha_adjustment}")
        print("---------------------------")
    
    def _save_checkpoint(self):
        """Save current optimizer state and session to a checkpoint file."""
        try:
            # Create checkpoint directory if it doesn't exist
            if not self.checkpoint_dir:
                self.checkpoint_dir = "checkpoints"
                Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
                
            # Generate filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_manual_{timestamp}.npz")
            
            # Get session info
            session_info = self._get_session_info()
            
            # Save checkpoint
            save_checkpoint(self.optimizer, session_info, checkpoint_path)
            
            print(f"Checkpoint saved to {checkpoint_path}")
            return checkpoint_path
        except Exception as e:
            print(f"Error saving checkpoint: {e}")
            return None
    
    def _print_help(self):
        """Print help information."""
        print("\n--- Available Commands ---")
        print("pause:         Pause optimization")
        print("resume:        Resume optimization")
        print("stop:          Stop optimization")
        print("stats:         Display optimization statistics")
        print("save:          Save checkpoint")
        print("alpha <value>: Adjust learning rate multiplier")
        print("params:        Display optimizer parameters")
        print("help:          Display this help message")
        print("------------------------")
    
    @classmethod
    def load_session(cls, checkpoint_path, fitness_function):
        """
        Load optimizer and session from a checkpoint file.
        
        Args:
            checkpoint_path: Path to the checkpoint file
            fitness_function: Function to evaluate solutions
        
        Returns:
            InteractiveOptimizer: Loaded interactive optimizer
        """
        # Load checkpoint
        optimizer_state, session_info = load_checkpoint(checkpoint_path)
        
        if optimizer_state is None or session_info is None:
            raise ValueError(f"Failed to load checkpoint from {checkpoint_path}")
        
        # Determine optimizer type from state
        from pyevo.optimizers import SNES, CMA_ES, PSO
        
        # Create the appropriate optimizer
        if "C" in optimizer_state:  # CMA-ES has a covariance matrix
            optimizer = CMA_ES(solution_length=optimizer_state["solution_length"])
        elif "velocities" in optimizer_state:  # PSO has velocities
            optimizer = PSO(solution_length=optimizer_state["solution_length"])
        else:  # Default to SNES
            optimizer = SNES(solution_length=optimizer_state["solution_length"])
        
        # Set optimizer state
        for key, value in optimizer_state.items():
            if hasattr(optimizer, key):
                setattr(optimizer, key, value)
        
        # Create interactive optimizer
        interactive_opt = cls(optimizer, fitness_function)
        
        # Set session state
        if "current_iteration" in session_info:
            interactive_opt.current_iteration = session_info["current_iteration"]
        if "best_fitness" in session_info:
            interactive_opt.best_fitness = session_info["best_fitness"]
        if "iter_history" in session_info:
            interactive_opt.iter_history = session_info["iter_history"]
        if "fitness_history" in session_info:
            interactive_opt.fitness_history = session_info["fitness_history"]
        if "sigma_history" in session_info:
            interactive_opt.sigma_history = session_info["sigma_history"]
        if "alpha_adjustment" in session_info:
            interactive_opt.alpha_adjustment = session_info["alpha_adjustment"]
            
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Iteration: {interactive_opt.current_iteration}, Fitness: {interactive_opt.best_fitness:.6f}")
        
        return interactive_opt 