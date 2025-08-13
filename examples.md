# Examples
Here we give some of the examples from our paper.

## Uncontrolled Rayleigh Bénard
<!-- |  Uncontrolled flow $Ra=10^4$                     |                Uncontrolled flow $Ra=10^5$|
|:-----------------------------------:|:-----------------------------------:|
| ![alt text](videos/Ra1e4Ra1e4.gif) | ![alt text](videos/Ra1e4Ra1e4.gif) |
|  **Uncontrolled flow** $Ra=10^6$                     |                **Uncontrolled flow** $Ra=5 * 10^5$|
| ![alt text](videos/Ra1e4Ra1e4.gif) | ![alt text](videos/Ra1e4Ra1e4.gif) | -->

TODO: Add the uncontrolled scenarios to the file.

## Comparison of Domain-Informed Reinforcement Learning and Uninformed Reinforcement Learning
We make a compelling case for the inclusion of domain knowledge to efficiently obtain robust control of chaotic flows. On the left is the typical performance of a Domain-Informed agent on a random initial condition, and on the right is the typical performance of an uninformed agent. Althoug the both agents achieve a similar reduction of average convective heat transfer, note the large qualitative differences in the obtained flows:

- The domain-informed agent eventually achieves **almost steady flow and remains at this state**
- The uninformed agent possesses **considerably unsteady flow.**

|  Domain-Informed                   |                Uninformed|
|:-----------------------------------:|:-----------------------------------:|
| $Ra=10^4$ | $Ra=10^4$ |
| ![alt text](videos/Ra1e4.gif) | ![alt text](videos/Ra1e4_NoRS.gif) |
| $Ra=10^5$ | $Ra=10^5$ |
| ![alt text](videos/Ra1e5.gif) | ![alt text](videos/Ra1e5_NoRS.gif) |

We don't show here for even higher Ra. They still result in chaotic flows, although lowering the average Nusselt number, similar to the Uninformed case for $Ra=10^5$ above. This is likely due to the cell distance measurement breaking down for more chaotic flows, which can be made more robust through filtering techniques. We do have evidence that control at $Ra=10^6$ is possible in our generalization experiment below.

## Generalization to other flow regimes
Here we show how the agents (Domain-Informed and Uninformed) that were trained on a Rayleigh number of $Ra=10^5$ perform on higher and lower Rayleigh numbers.

|  Domain-Informed                    |               Uninformed|
|:-----------------------------------:|:-----------------------------------:|
| $Ra=10^4$ | $Ra=10^4$ |
| ![alt text](videos/Ra1e5Ra1e4.gif) | ![alt text](videos/Ra1e5Ra1e4_NoRS.gif) |
| $Ra=10^6$ | $Ra=10^6$ |
| ![alt text](videos/Ra1e5Ra1e6.gif) | ![alt text](videos/Ra1e5Ra1e6_NoRS.gif) |

