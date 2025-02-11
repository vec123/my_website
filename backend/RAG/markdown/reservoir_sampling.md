# Reservoir Sampling

Reservoir Sampling is an algorithm to sample from a potentially infinite stream $\Sigma$. Each element in the sample has the same uniform probability of being picked and the sample has size $M$ always. Getting samples from infinite (or very large) streams has several applications. For example, sampling websites from the net, sampling metro passengers, sampling visitors, and many more. The context in which I learned about this algorithm was related to sampling edges from a graph with many edges — many more than fit into RAM. To estimate the number of triangles (or other motifs) in the graph, one needs to look at the edges. If one cannot look at all of them, at least some are required. Each edge must be sampled with equal probabilities.

The algorithm goes as follows:

## Algorithm

  1. Initialize
     * $t \leftarrow 0$
     * $S \leftarrow \emptyset$



For each $x_t \in \Sigma$:

  1. $t \leftarrow t + 1$
  2. If $t \leq M$:
     * $S \leftarrow S \cup x_t$
     * Else:
       * If $SampleElement\left(\frac{M}{t}\right)$:
         * Pick $x_i$ from $S$ uniformly at random (probability $\frac{1}{M}$)
         * $S \leftarrow S \setminus \\{x_i\\}$
         * $S \leftarrow S \cup \\{x\\}$



## Proposition

**Proposition:**

  1. **i)** $\Pr(x_i \in S) = \frac{M}{t}$ for all $i$ and $t$



**Proof by induction:**

$$ t = M \Rightarrow \Pr(x \in S) = 1 $$ 

Now $t \geq M$:

$$ \Pr(x_i \in S ) = \Pr(x_i \in S \text{ and } A ) + \Pr(x_i \in S \text{ and } \overline{A}) $$ 

and:

$$ \Pr(x_i \in S \text{ and } A ) = \Pr(x_i \in S \mid A ) \Pr(A) $$ $$ \Pr(x_i \in S \text{ and } \overline{A} ) = \Pr(x_i \in S \mid \overline{A} ) \Pr(\overline{A}) $$ 

Note:

$$ \Pr(A) = \frac{M}{t} $$ 

This is the probability that the element $x_i$ is removed and $x_t$ is added to the samples. $\Pr(\overline{A}) = 1- \Pr(A)$.

$$ \Pr(x_i \in S \mid A ) = \frac{M}{t-1} \left(1-\frac{1}{M}\right), $$ 

where $\frac{M}{t-1}$ comes by induction and $1-\frac{1}{M}$ is the probability that $x_i$ was not removed due to event $A$.

$$ \Pr(x_i \in S \mid \overline{A} ) = \frac{M}{t-1}. $$ 

Thus:

$$ \Pr(x_i \in S ) = \frac{M}{t-1} \left(1-\frac{1}{M}\right)\frac{M}{t} + \frac{M}{t-1}\left(1-\frac{M}{t}\right) = \frac{M}{t}. $$ 

It is trivial that $|S| = M$ for all $t$.

