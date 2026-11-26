# Implementation notes

Properties of the code that matter when interpreting or extending the results.

## Split model

All methods use the same ResNet-18 for 32×32 inputs, cut before `layer2`.  The client
segment holds the stem and the first residual stage with GroupNorm (149,824 parameters); the
server segment holds the rest with BatchNorm (11,024,138 parameters).

`split_model` moves leaf modules into a sequential client segment and leaves `nn.Identity`
behind.  This has two consequences, which apply to every method and every result:

- **The client segment has no skip connections.**  It applies the four convolutions of the first
  stage in sequence, without the two residual additions and the ReLU that follows each of them.
- **The server multiplies the cut activation by four.**  The two emptied residual blocks remain
  in the server segment, and each computes `x + x`.

The network is therefore not numerically equal to an unsplit ResNet-18.  The comparisons between
methods are unaffected, because all of them use this model.  `tests/test_models_and_step.py`
checks the behavior.

## One model represents all client replicas of a workload

Every client replica of a workload applies the same summed gradient and stays identical to the
others.  The code keeps one client model per workload; a test compares it with explicit
per-client replicas and optimizers.

The schedule-accounted trainer runs the client segment once per batch (`batch_clients: true`).
The real-execution engine runs it once per active client, so its step time grows with the number
of active clients.  Both give the same update up to rounding.

## Metrics

All accuracies are standard accuracy: correct predictions divided by the number of test examples.

## Randomness

- Real execution: every random stream has its own SHA-256-derived seed, so a run is determined by
  its configuration and seed on fixed hardware and library versions.  Results do not depend on the
  number of data-loader workers.
- Schedule-accounted: seeded in the same way.

## Time to target (real execution)

The time covers data setup, model initialization, client clustering, training, collectives,
barrier waits and all validation passes up to the crossing.  One warm-up update before the first
epoch is excluded.  Initialization takes 17 s for label-aware GCPSL, almost all of it for
clustering, and 0.2 s for the other methods.

The runs of the paper also scanned all parameters, gradients and optimizer states for non-finite
values around every optimizer step.  This code only checks the loss, so its epochs are somewhat
faster.  Results are unchanged.

## Sequential control

`gcpsl_label_aware_sequential` lets the workers take turns within every system phase: each worker
runs one local step while the others wait.  In the schedule-accounted studies, sequential
execution finishes one cluster before starting the next.  Epoch fusion makes both orders
equivalent for learning.

## Fusion

Replicas are averaged in float64 with sample-count weights and cast back to FP32.
`fusion: 100` fuses every 100 local steps, summed over all clusters, and additionally at the end
of every epoch.

## Participation and inactivity

- Real execution counts client–phase slots, including phases in which a cluster waits at the
  barrier.
- Schedule-accounted inactivity is averaged within each cluster's own rounds and does not count
  barrier waiting.

## Assignment rules

Label-aware clustering never moves a client into a cluster beyond 1.25 times the mean cluster
size.  Size-balanced clustering falls back to the lightest cluster if no cluster can take a client
within the cap; this does not happen in the paper's configurations.  The label-aware search
evaluates every single-client move in Python and takes minutes for CIFAR-100 and for N=16.

## Data augmentation

Random crops use a padding of 1 pixel for CIFAR-10 and 4 pixels for CIFAR-100.

## Payload and memory

- *Logical application payload* (real execution) counts every tensor transfer once.
- *Analytical payload* (schedule-accounted) counts every transfer at both endpoints and the client
  model exchange with all clients in every round.  The two must not be compared.
- *Synchronization volume* counts the tensors passed to collectives.  Fusion reduces in float64,
  so the reduced tensors count eight bytes per value.
- *Peak memory* in `results/schedule_accounted/peak_memory.csv` is the device memory in use,
  sampled with `nvidia-smi`, not the memory allocated by PyTorch.

## Batch deviation

The CIFAR-10 values average the deviation over all batches.  The CIFAR-100 values
(`logged_batch_deviation`) were logged at a lower frequency.
