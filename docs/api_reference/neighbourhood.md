# Neighbourhood Rule

Defines the neighborhood function \( N(x) \) for a given solution \( x \).

The neighborhood \( N(x) \) is defined as:

$$
N(x) = \{ y(x, p_1, p_2, c_1, c_2) : p_1 = 0, \dots, 50, \, p_2 = 0, \dots, 50, \, p_1 \neq p_2, \, c_1 \neq c_2 \}
$$

Where:

- \( p_1 \): Index of the first pot.
- \( p_2 \): Index of the second pot.
- \( c_1 \): Index of the first crucible.
- \( c_2 \): Index of the second crucible.
- \( x \): The current solution.

Each neighbor \( y(x, p_1, p_2, c_1, c_2) \) is defined as:
$$
y(x, p_1, p_2, c_1, c_2) = \{ y_1, y_2, \dots, y_{17} \}
$$

For each element \( y_i \) in the neighbor:
$$
y_i =
\begin{cases} 
x_{p_1}, & \text{if } i = p_1 \text{ and } c_1 \neq c_2, \\
x_{p_2}, & \text{if } i = p_2 \text{ and } c_1 \neq c_2, \\
x_i, & \text{otherwise.}
\end{cases}
$$

::: neighbourhood_rule