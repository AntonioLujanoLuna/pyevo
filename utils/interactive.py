"""
Interactive Mode for SNES Optimization

This module provides interactive control functionality for SNES optimization,
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

class InteractiveOptimizer:
    """
    Interactive wrapper for SNES optimizer that provides runtime control.
    
    Features:
    - Pause/resume optimization
    - Adjust parameters during runtime
    - Real-time statistics
    - Checkpointing
    """
    
    def __init__(self, optimizer, fitness_function, max_iterations=None, checkpoint_dir=None):
        """
        Initialize the interactive optimizer.
        
        Args:
            optimizer: SNES optimizer instance
            fitness_function: Function to evaluate solutions
            max_iterations: Maximum number of iterations (default: None = unlimited)
            checkpoint_dir: Directory for automatic checkpoints (default: None = no auto checkpoints)
        """
        self.optimizer = optimizer
        self.fitness_function = fitness_function
        self.max_iterations = max_iterations
        self.checkpoint_dir = checkpoint_dir
        
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
                    self.optimizer.save_state(checkpoint_path)
                
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
        self.sigma_history.append(float(self.optimizer.get_stats()["sigma_mean"]))
        
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
        """Print current progress of the optimization."""
        elapsed = time.time() - self.start_time
        stats = self.optimizer.get_stats()
        
        print(f"Iteration {self.current_iteration}: Best fitness = {self.best_fitness:.6f}, "
              f"σ_mean = {stats['sigma_mean']:.6e}, "
              f"Time: {elapsed:.1f}s")
    
    def _print_stats(self):
        """Print detailed statistics about the optimization."""
        elapsed = time.time() - self.start_time
        stats = self.optimizer.get_stats()
        
        print("\n--- Optimization Statistics ---")
        print(f"Iterations: {self.current_iteration}")
        print(f"Best fitness: {self.best_fitness:.6f}")
        print(f"Center mean: {stats['center_mean']:.6f}")
        print(f"Sigma mean: {stats['sigma_mean']:.6e}")
        print(f"Elapsed time: {elapsed:.1f} seconds")
        print("------------------------------\n")
    
    def _print_params(self):
        """Print current parameters of the optimizer."""
        print("\n--- Optimizer Parameters ---")
        print(f"Solution length: {self.optimizer.solution_length}")
        print(f"Population count: {self.optimizer.population_count}")
        print(f"Learning rate adjustment: {self.alpha_adjustment}")
        print("--------------------------\n")
    
    def _save_checkpoint(self):
        """Save a checkpoint of the current optimizer state."""
        timestamp = int(time.time())
        checkpoint_path = f"checkpoint_{timestamp}.npz"
        
        if self.checkpoint_dir:
            checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_path)
            
        self.optimizer.save_state(checkpoint_path)
        
        # Save additional interactive session info
        info_path = checkpoint_path.replace('.npz', '_info.json')
        info = {
            'iteration': self.current_iteration,
            'best_fitness': float(self.best_fitness),
            'elapsed_time': time.time() - self.start_time,
            'alpha_adjustment': self.alpha_adjustment
        }
        
        with open(info_path, 'w') as f:
            json.dump(info, f)
        
        print(f"Checkpoint saved to {checkpoint_path}")
        
    def _print_help(self):
        """Print help information for available commands."""
        print("\n--- Available Commands ---")
        print("pause - Pause the optimization")
        print("resume - Resume the optimization")
        print("stop - Stop the optimization")
        print("stats - Show detailed statistics")
        print("save - Save a checkpoint of the current state")
        print("alpha <value> - Adjust the learning rate multiplier")
        print("params - Show current optimizer parameters")
        print("help - Show this help message")
        print("------------------------\n")
        
    @classmethod
    def load_session(cls, checkpoint_path, fitness_function):
        """
        Load an interactive optimization session from a checkpoint.
        
        Args:
            checkpoint_path: Path to the optimizer checkpoint file
            fitness_function: Function to evaluate solutions
            
        Returns:
            InteractiveOptimizer instance with restored state
        """
        # Load optimizer state
        from snes import SNES
        optimizer = SNES.load_state(checkpoint_path)
        
        # Create interactive optimizer
        interactive = cls(optimizer, fitness_function)
        
        # Try to load additional session info
        info_path = checkpoint_path.replace('.npz', '_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                info = json.load(f)
                
            interactive.current_iteration = info.get('iteration', 0)
            interactive.best_fitness = info.get('best_fitness', float('-inf'))
            interactive.elapsed_time = info.get('elapsed_time', 0)
            interactive.alpha_adjustment = info.get('alpha_adjustment', 1.0)
        
        print(f"Session loaded from {checkpoint_path}")
        print(f"Current iteration: {interactive.current_iteration}")
        
        return interactive


def run_interactive_example():
    """Example showing how to use the interactive optimizer."""
    import numpy as np
    from snes import SNES
    
    # Define a simple objective function (minimize x^2)
    def objective(x):
        return -np.sum(x**2)  # Negative because SNES maximizes
    
    # Create optimizer for a 10-dimensional problem
    optimizer = SNES(solution_length=10)
    
    # Create interactive optimizer
    interactive = InteractiveOptimizer(
        optimizer=optimizer,
        fitness_function=objective,
        max_iterations=1000,
        checkpoint_dir="checkpoints"
    )
    
    # Start interactive optimization
    best_solution, best_fitness = interactive.start()
    
    print(f"Optimization complete.")
    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")


if __name__ == "__main__":
    run_interactive_example() 