# Examples
Here we give some of the examples from our paper.

## Uncontrolled Rayleigh Bénard
|  Uncontrolled flow $Ra=10^4$                     |                Uncontrolled flow $Ra=10^5$|
|:-----------------------------------:|:-----------------------------------:|
| ![alt text](videos/uncontrolled_Ra1e4.gif) | ![alt text](videos/uncontrolled_Ra1e5.gif) |
|  **Uncontrolled flow** $Ra=10^6$                     |                **Uncontrolled flow** $Ra=5 * 10^5$|
| ![alt text](videos/uncontrolled_Ra1e6.gif) | ![alt text](videos/uncontrolled_Ra51e6.gif) |


## Experiment 1 and 2 (Flow control of RBC) Comparison of Domain-Informed Reinforcement Learning and Uninformed Reinforcement Learning
We make a compelling case for the inclusion of domain knowledge to efficiently obtain robust control of chaotic flows. On the left is the typical performance of a Domain-Informed agent on a random initial condition, and on the right is the typical performance of an uninformed agent. Althoug the both agents achieve a similar reduction of average convective heat transfer, note the large qualitative differences in the obtained flows:

- The domain-informed agent eventually achieves **almost steady flow and remains at this state**
- The uninformed agent possesses **considerably unsteady flow.**

|  Domain-Informed                   |                Uninformed|
|:-----------------------------------:|:-----------------------------------:|
| $Ra=10^4$ | $Ra=10^4$ |
| ![alt text](videos/Ra1e4.gif) | ![alt text](videos/Ra1e4_NoRS.gif) |
| $Ra=10^5$ | $Ra=10^5$ |
| ![alt text](videos/Ra1e5.gif) | ![alt text](videos/Ra1e5_NoRS.gif) |
| $Ra=10^6$ | $Ra=10^6$ |
| ![alt text](videos/Ra1e6.gif) | ![alt text](videos/Ra1e6_NoRS.gif) |

As can be seen, the domain-informed control is very successful in the regime $Ra=10^5$, where it transforms a flow that is originally chaotic in a steady flow with one Bénard cell. This behavior is consistent over different initial conditions and the achieved one-cell state remains stable.
For $Ra=10^6$, the RL agent fails to merge the cells, which probably is due to the extra noise so that the cell distance measurement becomes unreliable. However, we do have successful cell merging events for this case, but the subsequent Nusselt number reduction does not yield a steady flow. See the generalization experiment below where the agent successfully merges cells for $Ra=10^6$, but the subsequent flow remains unsteady. Hence, this regime is intrinsically chaotic.
The picture for $Ra=5*10^6$ is very similar to that of $Ra=10^6$, and is therefore omitted here.

## Experiment 3: Generalization to other flow regimes
Here we show how the agents (Domain-Informed and Uninformed) that were trained on a Rayleigh number of $Ra=10^5$ perform on higher and lower Rayleigh numbers.

|  Domain-Informed                    |               Uninformed|
|:-----------------------------------:|:-----------------------------------:|
| $Ra=10^4$ | $Ra=10^4$ |
| ![alt text](videos/Ra1e5Ra1e4.gif) | ![alt text](videos/Ra1e5Ra1e4_NoRS.gif) |
| $Ra=10^6$ | $Ra=10^6$ |
| ![alt text](videos/Ra1e5Ra1e6.gif) | ![alt text](videos/Ra1e5Ra1e6_NoRS.gif) |

