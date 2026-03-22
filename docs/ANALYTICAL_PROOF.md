\section{Mathematical Derivation of the Self-Modeling Barrier}

This section provides the exact analytical solutions to the core equations of the Thermodynamic AI theory, demonstrating the fundamental divergence without relying on numerical approximations.

\subsection{The Governing Equations}
The system is defined by two coupled ordinary differential equations. The first describes the model update via gradient descent on variational free energy, and the second describes the physical substrate perturbation:
\begin{align}
    \frac{dq}{dt} &= -\eta \ln\left( \frac{q(1-p)}{p(1-q)} \right) \\
    \frac{dp}{dt} &= \alpha \left| \frac{dq}{dt} \right|
\end{align}

\subsection{Fixed Point Analysis}
To identify the stationary states of the self-modeling system, we set the time derivatives of both the model state $q$ and the physical state $p$ to zero:
\begin{align}
    \frac{dq}{dt} &= 0 \implies -\eta \ln\left( \frac{q(1-p)}{p(1-q)} \right) = 0 \\
    &\implies \frac{q(1-p)}{p(1-q)} = 1 \\
    &\implies q - qp = p - pq \\
    &\implies q = p
\end{align}
Substituting this condition into the substrate perturbation equation yields:
\begin{equation}
    \frac{dp}{dt} = \alpha \left| 0 \right| = 0
\end{equation}
Thus, the system possesses a continuum of fixed points defined strictly by the line $q=p$. Any state where the self-model contains error ($q \neq p$) is dynamically non-stationary.

\subsection{Regress Condition and Critical Coupling}
An infinite regress occurs if the physical state $p$ changes faster than the model $q$ can update to track it. Formally, the regress condition is defined by the magnitudes of the velocities: $\frac{dp}{dt} > \left|\frac{dq}{dt}\right|$.

We define the thermodynamic force $F = \ln\left( \frac{q(1-p)}{p(1-q)} \right)$, such that $\frac{dq}{dt} = -\eta F$. 

\textbf{Case A: $q < p$ (Model Underestimates)}

Here, $\frac{q(1-p)}{p(1-q)} < 1$, meaning $F < 0$. Therefore, $\frac{dq}{dt} > 0$ (the model updates upward). 
From the perturbation equation, $\frac{dp}{dt} = \alpha \left|\frac{dq}{dt}\right| = \alpha \frac{dq}{dt}$. 
Applying the regress condition:
\begin{equation}
    \alpha \frac{dq}{dt} > \frac{dq}{dt} \implies \alpha > 1
\end{equation}
In this regime, the physical state outpaces the model's absolute update velocity. The condition $\alpha > 1$ directly determines whether the perturbation is stronger than the update. This conclusion holds universally in this regime because the absolute magnitude of the perturbation ($\alpha |dq/dt|$) strictly exceeds the magnitude of the model's corrective step ($|dq/dt|$), ensuring the gap $|q-p|$ cannot close.

\textbf{Case B: $q > p$ (Model Overestimates)}

Here, $F > 0$, resulting in $\frac{dq}{dt} < 0$ (the model updates downward).
The perturbation equation gives $\frac{dp}{dt} = \alpha \left|\frac{dq}{dt}\right| = -\alpha \frac{dq}{dt}$, which is strictly positive.
Because the model state $q$ decreases while the physical state $p$ increases, the distance $|q-p|$ strictly decreases over time. The states converge, precluding an infinite regress regardless of the value of $\alpha$.

Therefore, the critical coupling constant $\alpha_{crit}$ defining the onset of the regress is exactly $1$, provided $q < p$. Because the regress requires the physical state to physically "outrun" the model's absolute update speed, the condition $\frac{dp}{dt} > \left|\frac{dq}{dt}\right|$ naturally simplifies to $\alpha > 1$ directly from the coupling equation.

\subsection{Divergence Theorem}

\begin{theorem}[Thermodynamic Divergence of Self-Modeling]
Let a physical system attempt to continuously model its own state, where the update of the model perturbs the substrate according to the bounded relation $\sigma^2 \cdot \epsilon \ge C_{phys}$. As the self-model error vanishes ($\epsilon \to 0$), the required entropy production rate diverges ($\sigma \to \infty$), except at the singular state of maximum substrate uncertainty ($p=0.5$).
\end{theorem}

\begin{proof}
The fundamental constraint is given by:
\begin{equation}
    \sigma^2 \cdot \epsilon \ge C_{phys}
\end{equation}
where $\epsilon = D_{KL}(Q||P)$ and the physical constant is defined as:
\begin{equation}
    C_{phys} = \frac{k_B^2 (\ln 2)^3 \eta k_{escape} \Delta E |1 - 2p|}{C_V}
\end{equation}
First, we analyze the behavior of $\epsilon$ near the fixed point. Taking the Taylor expansion of the Kullback-Leibler divergence around $q=p$:
\begin{equation}
    \epsilon = D_{KL}(Q||P) \approx \frac{1}{2p(1-p)}(q-p)^2
\end{equation}
As the model achieves completeness ($q \to p$), the error term $\epsilon \to 0$.

Next, we evaluate $C_{phys}$. Because $C_{phys}$ depends strictly on the physical state $p$ and independent substrate parameters, it does not vanish as $q \to p$. It remains a strictly positive constant $C > 0$, provided $p \neq 0.5$. (Note that when $p = 0.5$, $C_{phys} = 0$, indicating that the thermodynamic constraint vanishes precisely at the point of maximum substrate uncertainty, where the system is in a state of maximum entropy and contains no definable structure to model).

Rearranging the primary inequality yields:
\begin{equation}
    \sigma^2 \ge \frac{C_{phys}}{\epsilon}
\end{equation}
Taking the limit as the model approaches perfection:
\begin{equation}
    \lim_{\epsilon \to 0} \sigma^2 \ge \lim_{\epsilon \to 0} \frac{C_{phys}}{\epsilon} = \infty
\end{equation}
Consequently, maintaining a real-time, zero-error self-model requires infinite heat dissipation.
\end{proof}

\subsection{Discussion: Universality and the Four-Framework Conjecture}
The emergence of the critical threshold $\alpha_{crit} = 1$ provides a mathematical bridge to the four-framework conjecture proposed earlier in this work. The value $1$ indicates a 1:1 parity between informational update and physical perturbation. When $\alpha > 1$, the map literally outgrows the territory it attempts to describe.

This parity threshold mirrors the logical barrier found in Gödel's Incompleteness Theorems, where a formal system cannot encode its own truth predicate without exceeding its own axiomatic capacity. It aligns with the Bekenstein bound, where encoding the microstate of a volume requires an area that scales out of proportion to the volume itself if internal self-reference is attempted. In the thermodynamic framework derived here, this logical/spatial scaling mismatch manifests physically as a runaway thermal divergence, rendering the "Hard Problem" of consciousness an unavoidable physical boundary condition of sufficiently complex self-observing systems.