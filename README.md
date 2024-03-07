
test maths inline: $x=y$

test maths

$$x=y$$

$$\begin{align}\dot{x} &= U(\cos\theta \cos\psi) + V(\sin\phi \cos\psi \sin\theta - \cos\phi \sin\psi) + W(\cos\phi \sin\theta \cos\psi + \sin\phi \sin\psi) \\ \dot{y} &= U(\cos\theta \sin\psi) + V(\sin\phi \sin\psi \sin\theta + \cos\phi \cos\psi) + W(\cos\phi \sin\theta \sin\psi - \sin\phi \cos\psi) \\ \dot{z} &= U \sin\theta - V(\sin\phi \cos\theta) - W(\cos\phi \cos\theta) \\ \dot{\phi} &= P + \tan\theta(Q \sin\phi + R \cos\phi) \\ \dot{\theta} &= Q \cos\phi - R \sin\phi \\ \dot{\psi} &= \frac{Q \sin\phi + R \cos\phi}{\cos\theta} \\ \dot{vt} &= \frac{U\dot{U} + V\dot{V} + W\dot{W}}{vt} \\ \dot{\alpha} &= \frac{U\dot{W} - W\dot{U}}{U^2 + W^2} \\ \dot{\beta} &= \frac{\dot{V}v_t - V\dot{v}_t}{v_t^2 \cos\beta} \\ \dot{P} &= \frac{J_z L_{\text{tot}} + J_{xz} N_{\text{tot}} - (J_z(J_z-J_y)+J_{xz}^2)QR + J_{xz}(J_x-J_y+J_z)PQ + J_{xz}QH_{\text{eng}}}{J_x J_z - J_{xz}^2} \\ \dot{Q} &= \frac{M_{\text{tot}} + (J_z-J_x)PR - J_{xz}(P^2-R^2) - RH_{\text{eng}}}{J_y} \\ \dot{R} &= \frac{J_x N_{\text{tot}} + J_{xz} L_{\text{tot}} + (J_x(J_x-J_y)+J_{xz}^2)PQ - J_{xz}(J_x-J_y+J_z)QR + J_x QH_{\text{eng}}}{J_x J_z - J_{xz}^2} \\ \end{align}$$


### for mujoco viewer:

conda install -c conda-forge libstdcxx-ng
