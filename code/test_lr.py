import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from transformers import get_linear_schedule_with_warmup

# Example model and optimizer
model = torch.nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)  # Base learning rate is 0.1
num_steps = 100    # Total number of steps for interpolation

# Define the LinearLR scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_training_steps=num_steps,
    num_warmup_steps=num_steps//10
)

# Track learning rates for plotting
lr_values = []

for step in range(num_steps):
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    lr_values.append(current_lr)

# Plot the learning rate schedule
plt.figure(figsize=(8, 5))
plt.plot(range(1, num_steps + 1), lr_values, marker="o", linestyle="-")
plt.title("Linear Learning Rate Schedule")
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.grid(True)

# Save the plot
plot_filename = "linear_lr_schedule.png"
plt.savefig(plot_filename, dpi=300)
plt.show()

print(f"Plot saved as '{plot_filename}'")
