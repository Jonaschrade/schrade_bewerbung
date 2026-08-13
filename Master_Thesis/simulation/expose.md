
\section{Motivation}
The general scope of the thesis revolves around polarization as a phenomenon of collective behavior among Large Language Model (LLM) agents in a social network graph. This topic carries both internal and external relevance. Internally, as pre-print platforms see a recent surge of studies investigating the dynamics and outcomes of collectively interacting AI agents, there is a particular need for contributions grounded in classical agent-based modelling and network theory that extend, rather than replace, established approaches. Externally, LLM-supplemented network simulation bears relevance beyond the academic sphere, since structural polarization remains a phenomenon of considerable societal concern whose underlying mechanisms warrant continued investigation using state-of-the-art methods. \\
Accordingly, this thesis pursues two central goals. First, it offers an innovative, dynamic extension of the classical agent-based model of Social Feedback Theory (SFT; \cite{Banisch2019}) by introducing LLM agents and network rewiring, thereby advancing understanding of the mechanisms of social-feedback-driven polarization and the societal implications that follow. Second, it contributes to the growing body of LLM behavioral research by shedding light on the black-box nature of these models and its implications for their collective behavior.

\section{Background}
The term Agent-Based Modeling (ABM) denotes a process-oriented, bottom-up simulation method in which populations of autonomous agents follow simple, locally specified decision rules, and from whose interactions higher-level patterns emerge that no single agent intends or perceives. In doing so, the approach preserves individual variation, spatial or network structure, and path dependence. It has thus become central to social-behavioral research, since ABM studies show how, and under what structural conditions, collective patterns arise from individual behavior rather than merely describing an aggregate end-state (\cite{Flache2017}; \cite{Goldstone2015}). \\

As such, SFT reflects a conceptual branch of ABM that was first propopsed by \textcite{Banisch2019}, with further enrichments by \textcite{Banisch2022}, \textcite{Jacob2023}, and \textcite{Banisch2026}. The framework models opinion expression as a reinforcement learning (RL) problem in which agents hold internal Q-values for two opinion stances (in favor / against). Agents express the higher-valued stance to their neighbors, receiving agreement or disagreement as a reward signal depending on each neighbor's own preferred stance. Q-values are updated iteratively across expression rounds until the system converges to an equilibrium. Although deliberately parsimonious in its assumptions, this mechanism generates stable bi-polarization, spiral-of-silence, and echo-chamber dynamics without invoking the assumptions of classical polarization models, such as negative influence, bounded confidence, or opinion homophily.

Furthermore, SFT provides an explicit, defensible micro-macro link grounded in social neuroscience at the micro level and mean-field and equilibrium analysis at the macro level, which makes it possible to isolate the structural effects of the network: \textcite{Banisch2019} show that connectivity drives consensus, whereas polarization requires sufficient community structure. Subsequent work demonstrates how the same reward-learning core, combined with different structural overlays, reproduces a range of empirically observed collective phenomena, such as the spiral of silence (\cite{Banisch2022}) and platform segregation (\cite{Banisch2026}).

While these results strengthen the model's internal validity, the parsimony of its underlying assumptions remains a core limitation of the Banisch tradition. Binary opinion stances, homogeneous rewards, and single scalar feedback serve simplistic modelling purposes well but arguably fail to capture the diversity and gradation of human cognition and communication, thereby limiting the external validity of these results when transferred to real-world applications.

This argument is supported by \textcite{Sarkozy2022}, who provide experimental evidence for SFT's claims, demonstrating that social feedback can influence not only public expression but even privately held opinions. Notably, whereas classical SFT treats feedback as a single summand within the reward function, they find that ambivalent social signals are the most influential drivers of opinion change, prompting respondents to shift toward stronger disagreement. This directly contradicts the SFT prediction that mixed feedback drives agents toward moderation by equalizing the Q-values of competing stances.

The limitations of SFT reflect a broader issue: the generalizability of ABMs is constrained by design. Impoverished agent cognition, weak empirical calibration, and limited cross-model comparability have prompted calls across the field for richer agent architectures and tighter coupling between simulation and experiment (\cite{Flache2017};~\cite{Goldstone2015})--calls that recent developments in the LLM research community appear to heed.

Since \textcite{Park2023} generative-agents architecture first demonstrated that LLMs equipped with a memory-reflection-planning cognition could sustain coherent multi-agent social dynamics, the literature integrating LLMs into ABMs has expanded rapidly. The body of work that followed has shown that LLM-based agents can spontaneously reproduce fundamental social phenomena without these being explicitly programmed. Across this literature, such phenomena include coordination and (mis-)alignment (~\cite{Bellina2026};~\cite{DeMarzo2025}), endogenous network formation (\cite{DeMarzo2023};~\cite{Schneider2025}), (mis-)information and belief propagation (\cite{Chuang2024};~\cite{Liu2024}), the emergence of shared norms and conventions (~\cite{Ashery2025};~\cite{Cordova2024}), and homophily, polarization, and echo chambers (\cite{Donkers2025};~\cite{Ferraro2024};~\cite{Ohagi2024};~\cite{Piao2026};~\cite{Sakurai2025};~\cite{Wang2026}).

As LLM agents replace hand-crafted update rules with natural-language reasoning, persona-conditioned behavior, and richly contextualized interaction, the LLM-ABM paradigm relaxes several of the most restrictive assumptions on which opinion-dynamics models and classical ABMs have long depended. Yet this same flexibility produces a methodological tension: the mechanism driving any given opinion update resides within the LLM forward pass and is therefore largely opaque. Emergent macro-patterns can be observed but not readily attributed to identifiable micro-processes, leaving the field with rich phenomenology but limited mechanistic explanation.

This is precisely where a methodological synthesis offers a productive complement. By providing an explicit, parsimonious, and analytically tractable micro-mechanism, namely reward-driven reinforcement learning over expressed opinion stances, together with an established link from individual learning to collective regimes, SFT supplies exactly the mechanistic traceability that LLM-ABMs currently lack. LLM-based agents, in turn, contribute the natural-language expression, graded and ambivalent feedback, and heterogeneous personas absent from SFT in its canonical form. Integrating the two addresses both sides' core limitations, grounding the LLM-ABM paradigm in falsifiable opinion-dynamics theory while extending SFT beyond the binary, homogeneous-reward setting of its original formulation and assessment.

\section{Proposed Methods}
\subsection{Gap + Research Questions} 
The methodological gap can thus be characterized as follows. LLM-ABM reproduces social phenomena but lacks a falsifiable micro-mechanism, whereas SFT provides one but cannot accommodate natural-language expression, ambivalent feedback, or heterogeneous agent personas. From this tension, the \textbf{overall research question} is formulated:

\begin{quote}
Can Social Feedback Theory serve as underlying reinforcement-learning mechanism for LLM agents embedded in a structured social network such that the resulting hybrid model preserves SFT's analytic interpretability while extending the empirical scope?
\end{quote}

By mirroring the SFT of the Banisch canon as closely as possible, a tangible replication sub-question is derived.
\begin{quote}
    \textbf{RQ1:} On a static community-structured network, does SFT-governed LLM-ABM reproduce the theory's macro-dynamics, specifically modularity-driven phase transition between consensus and stable bi-polarization and do LLM-generated feedback signals yield same Q-value learning trajectories as canonical binary reward rule.
\end{quote}
It is important to note that the research body defining SFT treats the agentic network as exogenous and static, itself unaffected by the feedback mechanism and the resulting polarization dynamics. For additional empirical novelty, this work departs from that assumption. Following recent findings in the LLM-ABM literature on endogenous network formation (\cite{DeMarzo2023};~\cite{Schneider2025}) and spontaneous homophily and hub emergence (\cite{Piao2026}), the network is instead treated as a learned outcome. Agents learn not only what to express but also whom to interact with, with both processes governed by the same SFT reward signal. This empirical augmentation manifests as second sub-question:

\begin{quote}
\textbf{RQ2:} When the interaction graph itself is endogenous, i.e., tie formation and survival are governed by the same social-feedback reward signal that drives opinion expression, does the coupled (network x opinion) mechanism still produce SFT's regimes? Under what conditions does it generate structural patterns (homophily, hub emergence, community modularity) reported in recent LLM-ABM literature? 
\end{quote}


\subsection{Simulation}

The ABM simulation\footnote{Source code: \url{https://github.com/Jonaschrade/thesis/tree/main/simulation}} is initialized with a stochastic block model of $n$ LLM agents. Each agent holds an internal state comprising of \\
(1) a realistic persona prompt based on actual survey data of the German General Social Survey,\\
(2) an opinion state that implements Banisch's dual Q-value mechanism reflecting the expected reward of expressing an opinion in favor or against the current discussion topic, \\
(3) and a personal Chroma database of past interactions and reflections that is updated and queried in alignment with memory stream for generative agents as proposed by \textcite{Park2023}.\\

At the first discussion round, an expressing agent is drawn uniformly from the set of agents with at least one connecting edge. From his connected neighbors, a responder is also drawn uniformly. Using a softmax-function with inverse temperature $\beta$

\begin{equation} 
    P(o_i) = \frac{\exp(\beta * Q(o_i))}{\exp(\beta * Q(o_i))+\exp(\beta * Q(o_j))}
\end{equation}

the stances on the topic (in favor ($o_i$) or against ($o_j$)) are stochastically drawn for both discussing agents. Before the first interaction, the Q-values of all agents are zero and thus, expressed stances manifest completely at random. After that, deviances from the preferred and the expressed stance are regulated by temperature $\beta$. \\
The expresser is prompted to respond to an external moderator statement\footnote{After the first exchange, the expresser responds to the neighbor's last message of their previous interaction.} with his name, persona, relevant memories, and current stance on the topic injected. The responder reacts to expresser's utterance in the same way. The LLM -- for now, purposefully detached from agentic persona and memory stream -- is then prompted to label the responder's reply based on its (dis-)agreement with the expresser as reward value $r$ on a continous $[-1, 1]$ scale. The Q-value of the expressed stance is accordingly updated only for the expresser following the temporal-difference update 

\begin{equation}
    Q(o_i)_{t+1} \leftarrow (1-\alpha) * Q(o_i)_t + \alpha * r_t
\end{equation}
with learning rate $\alpha$ from \textcite{Banisch2019}\\

During each discussion round, a number of agents equal to the total network size is selected as expressers, yielding one expected interaction loop per agent per round. While the classical Banisch approach models this loop as a single expression followed by a single response, extending it to a longer, self-contained exchange, in which only the responder's final message is propagated to the reward evaluation, would be straightforward. Every $k$ discussion rounds, agents are prompted to reflect, implementing a memory retrieval and synthesis mechanism in the sense of \textcite{Park2023} and promising richer, more coherent, and more realistic agent behavior. The simulation terminates after $m$ discussion rounds. Personas, discussions, reflections, and per-round snapshots of the network and agent states are logged throughout.

\subsection{Analysis}
To assess the simulation outcome and answer above research questions by testing derived hypotheses, polarization metrics and opinion diffusion parameters on three levels are proposed as replication and extension of the Banisch canon.
\subsubsection{Micro / Individual}
\begin{itemize}
    \item \textbf{Individual Q-value trajectories}. Evolution and final state of the Q-value pair quantifies an agent's learned valuations of each opinion stance. Combined with the preferred/expressed opinion and the agent's conviction, as operationalized by the gap between the two Q-values, the metrics provide insight on whether agents are radicalising, oscillating, or settling.
    \item Extensions: \begin{itemize}
        \item \textbf{Expression as high-dimensional object}. Posts can be embedded and tracked over time, which allows for measurement of semantic drift throughout the simulation.
        \item \textbf{Received reward}. LLM-generated feedback is graded and ambivalent, variance and skewness of received rewards enable insights on the mixed feedback annomaly as suggested by \textcite{Sarkozy2022}.
        \item \textbf{Q-update predictability}. Extent to which an agent's next expression can be predicted from its Q-state alone. 
    \end{itemize}
\end{itemize}
\subsubsection{Meso / Local}
\begin{itemize}
    \item \textbf{Local opinion climate}. Fraction of in-group vs. out-group ties per agent as well as the local agreement rate, i.e. the proportion of an agent's interaction that yields positive feedback, quantify how an agent perceives its immediate neighborhood. Furthermore, \textcite{Jacob2023} turn to the incongruent-links percentage -- the share of ties connecting agents with opposing expressed opinions -- as operational distinction between structural and unstructural polarization.
    \item Extensions: \begin{itemize}
        \item \textbf{Local semantic coherence}. Within-neighborhood similarity of expressed opinions (via embedding distance) captures whether agents in the same community converge linguistically as well as positionally. 
        \item \textbf{Tie-level reward statistics}. Mean and variance of reward flows across edges provide descriptive metrics on local social feedback dynamics.
    \end{itemize}
\end{itemize}
\subsubsection{Macro / Global}
\begin{itemize}
    \item \textbf{Phase transition between consensus and stable bi-polarization}. Global metrics such as overall opinion dispersion and mean conviction distinguish network polarization from consesus, and neutral consensus from extemised consensus, respectively. The regime classification at convergence, captured by aligned or opposite average opinions within each community, and its behavior at the inter-group coupling threshold provides the main dependent variable of interest. 
    \item Extensions: See Section~\ref{sec:dynamic_graph}
\end{itemize}

\subsection{Extensions}
\subsubsection{Dynamic Graph}\label{sec:dynamic_graph}

In order to address RQ2, endogeneity needs to be introduced to the network structure as a function of the same SFT reward mechanism that also governs agent stances. This dynamic is implemented by adding two additional steps to the interaction loop:\\
\begin{enumerate}
    \item The \textbf{Edge Update} rule controls edge survival with a unilateral assessment of the communication channel, quantfied by each agent's perceived strength of the connection to its neighbor. At the beginning of the simulation, each edge is initialized with a strength of 1.0 for every agent. After the expresser received the social reward of the current interaction and updated its Q-values, the reward is mean-aggregated in a sliding window with the previous $j-1$ rewards received via the assessed edge.\\ 
    The strength of edge $(u,v)$ for agent $u$ is then updated using the formula \\
    \begin{equation}
        \text{Strength}(u,v)_{u,t+1} \leftarrow \text{Strength}(u,v)_{u,t} + \bar{r}_{[t-j-1, t]} * \delta\text{,}
    \end{equation}
    where $\bar{r}_{[t-j-1, t]}$ represents the average of the current and last $j-1$ received rewards and $\delta$ the valuation rate.\\
    As soon as strengh reaches zero for either of the connected agents, the communication channel is interrupted and the edge removed.

    \item To \textbf{Ensure Connectivity} isolated agents can be randomly matched to any other agent at the end of a discussion round. While this guarantees no agent is permanently excluded from future rounds, it remains to be decided whether such network behavior is empirically favourable. Furthermore, as of yet, there is no edge creation mechanism in place implying that network density is strictly non-increasing. Random or preferential attachement, triadic closure, or opinion homophily would be logical options.
\end{enumerate}
Once tie formation and dissolution are endogenous, degree distribution, modularity, and assortativity become primary outcome indicators for investigating the micro-macro-relationship between social feedback and structural polarization.
\subsubsection{Homophily}
\textcite{Jacob2023} introduce the concept of probabilistic homophily for selecting the responder from an expresser's set of neighbors. Empirically, the mechanism is rooted in humans' preference for like-minded interaction. The authors operationalise this phenomena as weighted probability of interaction\\
\begin{equation}
    w_j = \exp(-h*|\Delta Q_u - \Delta Q_v|)\text{,}
\end{equation}
where $h$ represents the homophily control parameter and $\Delta Q$ the gap in Q-values of both stances. \\ 
Whether to operationalize homophily as a weight function of the inherently private gap in Q-values, or instead implement a matching weight based on `perceived homophily', remains an open question for this undertaking.

\subsubsection{Symmetric Feedback}
In the Banisch canon, the feedback learning mechanism is asymmetric by design: although both agents convey their currently preferred stance, whether through expression or response, only the expressing agent updates its Q-values. Within a complete interaction round, in which both agents express their standpoint and receive feedback at least once, a symmetric feedback rule would be straightforward to implement, empirically plausible, and conducive to both Q-learning and edge valuation.

\section{Next Steps}

June-July:
\begin{itemize}
    \item Test and fine-tune simulation model by pairwise and network interactions. 
    \item Configuration of main simulation loop, set governing parameters such as number of interacting agents $n$, maximum number of discussion rounds $m$, or reflection interval $k$ as well as the hyperparameters comprising of the stance expression temperature $\beta$, the reward learning rate $\alpha$, the edge valuation rate $\delta$, and the homophily control $h$.
    \item Final run of main simulation.
\end{itemize}

July-August:

\begin{itemize}
    \item Polarization analyis on micro, meso, and macro level.
\end{itemize}

September-November:

\begin{itemize}
    \item Thesis writing. 
\end{itemize}

\printbibliography[]

\end{document}
