# #!/usr/bin/env python
# # coding: utf-8

get_ipython().run_line_magic('matplotlib', 'widget')

import numpy as np
import matplotlib.pyplot as plt

from qutip import *
from ipywidgets import interactive, FloatSlider, IntSlider, Checkbox, VBox, HBox, Button, Output
from IPython.display import HTML, display, clear_output
from matplotlib import animation, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors

def Plot_HO(tlist = np.linspace(0, 100, 100)):
    # Initialize the figure outside the simulation function
    fig, ax = plt.subplots(figsize=(10, 8), layout='tight')

    # Add individual checkboxes for controlling transparency for each plot
    base_alpha_checkbox = Checkbox(value=False, description=r"Hide damped HO")
    driving_alpha_checkbox = Checkbox(value=False, description=r"Hide Driving")
    kerr_alpha_checkbox = Checkbox(value=False, description=r"Hide Kerr")

    # # Add a loading label
    # loading_label = HTML("<h3 style='color: blue;'>Loading...</h3>")
    # loading_label.layout.display = 'none'  # Initially hidden

    # Create sliders
    N_slider = IntSlider(value=30, min=1, max=100, description=r'Dim($\mathcal{H}$)')
    n_th_a_slider = IntSlider(value=4, min=1, max=10, description=r'$\langle \hat{n}_{E}\rangle$')
    psi0_val_slider = IntSlider(value=9, min=0, max=15, description=r'$\rho_0$')
    kappa_slider = FloatSlider(value=0.01, min=0.0, max=0.5, step=0.01, description=r'$\gamma$')
    Omega_slider = FloatSlider(value=0.01, min=0.0, max=0.5, step=0.01, description=r'$\Omega$')
    k_slider = FloatSlider(value=0.05, min=0.0, max=0.2, step=0.01, description=r'$\kappa$')

    # Create checkboxes for driving and Kerr effects
    driving_checkbox = Checkbox(value=False, description="Include Driving")
    kerr_checkbox = Checkbox(value=False, description="Include Kerr")
    
    # Create checkboxes for steady state soln.
    steady_H0_checkbox = Checkbox(value=False, description="Show steady state soln. for HO")
    steady_H1_checkbox = Checkbox(value=False, description="Show steady state soln. for Driving")
    steady_H2_checkbox = Checkbox(value=False, description="Show steady state soln. for Kerr")

    # Create the submit button
    submit_button = Button(description='Run Simulation')

    # Function to simulate and update plot
    def simulate_on_submit(N=30, n_th_a=4, psi0_val=9, kappa=0.03, driving=False, kerr=False, Omega=0.0, k=0.0, base_alpha=False, driving_alpha=False, kerr_alpha=False, steady_H0=False, steady_H1=False, steady_H2=False):
        ax.clear()  # Clear the previous plot

        a = destroy(N)

        # Base Hamiltonian
        H0 = a.dag() * a
        psi0 = basis(N, psi0_val)

        # Collapse operators
        c_op_list = []
        rate = kappa * (1 + n_th_a)
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * a)  # Decay operators
        rate = kappa * n_th_a
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * a.dag())  # Excitation operators

        # Set alpha values based on checkboxes
        base_alpha = 0.3 if base_alpha_checkbox.value else 1.0
        driving_alpha = 0.3 if driving_alpha_checkbox.value else 1.0
        kerr_alpha = 0.3 if kerr_alpha_checkbox.value else 1.0

        # Simulate for base Hamiltonian (H0)
        opts = {'store_states': False}
        medata_base = mesolve(H0, psi0, tlist, c_op_list, [a.dag() * a], options=opts)

        # Plot for base Hamiltonian
        ax.plot(tlist, medata_base.expect[0], lw=2, label=r'$\langle n(t) \rangle$ HO', alpha=base_alpha, color='tab:blue')

        # If driving is enabled, add the driving term and simulate
        if driving:
            H1 = Omega * (a.dag() + a)
            H_with_driving = H0 + H1
            medata_driving = mesolve(H_with_driving, psi0, tlist, c_op_list, [a.dag() * a], options=opts)
            ax.plot(tlist, medata_driving.expect[0], lw=2, linestyle='--', label=r'$\langle n(t) \rangle$ Driving', alpha=driving_alpha, color='tab:orange')

        # If Kerr nonlinearity is enabled, add the Kerr term and simulate
        if kerr:
            H2 = k * (a.dag() * a * a.dag() * a)
            H_with_kerr = H0 + (H1 if driving else 0) + H2
            medata_kerr = mesolve(H_with_kerr, psi0, tlist, c_op_list, [a.dag() * a], options=opts)
            ax.plot(tlist, medata_kerr.expect[0], lw=2, linestyle=':', label=r'$\langle n(t) \rangle$ Kerr', alpha=kerr_alpha, color='tab:green')

        # Set up the plot
        ax.set_xlabel('Time', fontsize=24)
        ax.set_ylabel(r'$\langle n \rangle$', fontsize=24)
        ax.set_title('Evolution of Photon Number Expectation Value', fontsize=18)
        
        # find steady-state solution
        if steady_H0:
            final_state_H0 = steadystate(H0, c_op_list)
            fexpt_H0 = expect(a.dag() * a, final_state_H0)
            plt.axhline(y=fexpt_H0, linestyle=':', lw=1.5, label='Steady H1', color='tab:blue')
        if steady_H1 and driving:
            H1 = Omega * (a.dag() + a)
            final_state_H0_1 = steadystate(H0+H1, c_op_list)
            fexpt_H0_1 = expect(a.dag() * a, final_state_H0_1)
            plt.axhline(y=fexpt_H0_1, linestyle=':', lw=1.5, label='Steady H2', color='tab:orange')
        
        if steady_H2 and kerr:
            H2 = k * (a.dag() * a * a.dag() * a)
            final_state_H0_1_2 = steadystate(H0 + (H1 if driving else 0) + H2, c_op_list)
            fexpt_H0_1_2 = expect(a.dag() * a, final_state_H0_1_2)
            plt.axhline(y=fexpt_H0_1_2, linestyle=(2,(2,2)), lw=1.5, label='Steady H3', color='tab:green')
            
        # Move the legend to the right side
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_ylim(0,25)
        
        # Redraw the updated plot
        plt.draw()

    # Function to be called when the button is clicked
    def on_submit(b):
        submit_button.disabled = True  # Disable the submit button
        # loading_label.layout.display = 'block'  # Show loading label
        simulate_on_submit(N_slider.value, n_th_a_slider.value, psi0_val_slider.value, 
                           kappa_slider.value, driving_checkbox.value, kerr_checkbox.value, 
                           Omega_slider.value, k_slider.value, 
                           base_alpha_checkbox.value, driving_alpha_checkbox.value, kerr_alpha_checkbox.value,
                           steady_H0_checkbox.value,steady_H1_checkbox.value,steady_H2_checkbox.value)
        # loading_label.layout.display = 'none'  # Hide loading label
        submit_button.disabled = False  # Re-enable the submit button

    # Attach the function to the button
    submit_button.on_click(on_submit)

    # Arrange widgets in a box, include the transparency checkboxes
    controls_box = VBox([HBox([N_slider, n_th_a_slider]),
                         HBox([psi0_val_slider, kappa_slider]), 
                         HBox([driving_checkbox, Omega_slider]), 
                         HBox([kerr_checkbox, k_slider]), 
                         HBox([base_alpha_checkbox, steady_H0_checkbox]),
                         HBox([driving_alpha_checkbox,steady_H1_checkbox]),
                         HBox([kerr_alpha_checkbox,steady_H2_checkbox]), 
                         submit_button])  # Include the loading label in controls

    # Display controls
    display(controls_box)

    # Call the simulation function once to show the initial plot
    simulate_on_submit(N_slider.value, n_th_a_slider.value, psi0_val_slider.value, 
                       kappa_slider.value, driving_checkbox.value, kerr_checkbox.value, 
                       Omega_slider.value, k_slider.value, 
                       base_alpha_checkbox.value, driving_alpha_checkbox.value, kerr_alpha_checkbox.value,
                       steady_H0_checkbox.value,steady_H1_checkbox.value,steady_H2_checkbox.value)



def Plot_Fock(tlist=np.linspace(0, 100, 100)):
    
    # Create output widget
    output = Output()
    
    # Create sliders and widgets
    N_slider = IntSlider(value=30, min=1, max=100, description=r'Dim($\mathcal{H}$)')
    n_th_a_slider = IntSlider(value=4, min=1, max=10, description=r'$\langle \hat{n}_{E}\rangle$')
    psi0_val_slider = IntSlider(value=9, min=0, max=30, description=r'$\rho_0$')
    kappa_slider = FloatSlider(value=0.01, min=0.0, max=0.5, step=0.01, description=r'$\gamma$')
    Omega_slider = FloatSlider(value=0.01, min=0.0, max=0.5, step=0.01, description=r'$\Omega$')
    k_slider = FloatSlider(value=0.05, min=0.0, max=0.2, step=0.01, description=r'$\kappa$')

    # Create checkboxes for driving and Kerr effects
    driving_checkbox = Checkbox(value=False, description="Include Driving")
    kerr_checkbox = Checkbox(value=False, description="Include Kerr")
    
    # Create the submit button
    submit_button = Button(description='Run Simulation')
        
        
    # Function to simulate and update plot
    def simulate_on_submit(N=30, n_th_a=4, psi0_val=9, kappa=0.03, driving=False, kerr=False, Omega=0.0, k=0.0):
        # Create new figure and axes
        fig, ax = plt.subplots(figsize=(10, 8))
        
        a = destroy(N)

        # Base Hamiltonian
        H0 = a.dag() * a
        H1 = Omega * (a.dag() + a)
        H2 = k * (a.dag() * a * a.dag() * a)
        psi0 = basis(N, psi0_val)

        # Collapse operators
        c_op_list = []
        rate = kappa * (1 + n_th_a)
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * a)  # Decay operators
        rate = kappa * n_th_a
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * a.dag())  # Excitation operators

        opts = {'store_states': True}
        H = H0
        if driving:
            H += H1
        if kerr:
            H += H2

        # Solve and create the animation for Fock distribution
        medata = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a], options=opts)
        
        title = f"$\\langle \\hat{{n}}_E\\rangle = {n_th_a}$, $\\gamma = {kappa}$"
        if driving:
            title += f", $\\Omega = {Omega}$"
        if kerr:
            title += f", $\\kappa = {k}$"
        fig.suptitle(title)
        
        # Precompute probabilities
        probs = [np.real(np.diag(state.full())) for state in medata.states]
        n = np.arange(N)
        
        # Initialize the bar plot
        bars = ax.bar(n, probs[0], align='center',color='skyblue', edgecolor='black')
        ax.set_xlabel('Fock state $n$')
        ax.set_ylabel('Occupation probability')
        ax.set_xlim(-0.5, N - 0.5)
        ax.set_ylim(0, 1)
        
        def update(frame):
            for bar, height in zip(bars, probs[frame]):
                bar.set_height(height)
            return bars
        
        anim = animation.FuncAnimation(fig, update, frames=len(probs), blit=False, interval=100)
        
        plt.close(fig)  # Close the figure to prevent duplicate display
        
        # Display the animation
        html_anim = HTML(anim.to_jshtml())
        display(html_anim)
        

    # Function to be called when the button is clicked
    def on_submit(b):
        submit_button.disabled = True  # Disable the submit button
        with output:
            clear_output(wait=True)
            simulate_on_submit(N_slider.value, n_th_a_slider.value, psi0_val_slider.value, 
                               kappa_slider.value, driving_checkbox.value, kerr_checkbox.value, 
                               Omega_slider.value, k_slider.value)
        submit_button.disabled = False  # Re-enable the submit button
        


    # Attach the function to the button
    submit_button.on_click(on_submit)

    # Arrange widgets in a box and display controls only once
    controls_box = VBox([HBox([N_slider, n_th_a_slider]),
                        HBox([psi0_val_slider, kappa_slider]), 
                        HBox([driving_checkbox, Omega_slider]), 
                        HBox([kerr_checkbox, k_slider]), 
                        submit_button])

    display(controls_box)
    display(output)

    # Call the simulation function once to show the initial plot
    with output:
        simulate_on_submit(N_slider.value, n_th_a_slider.value, psi0_val_slider.value, 
                           kappa_slider.value, driving_checkbox.value, kerr_checkbox.value, 
                           Omega_slider.value, k_slider.value)

def Plot_Wigner(tlist=np.linspace(0, 100, 100), xvec=np.linspace(-5, 5, 500)):
    # Create output widget
    output = Output()

    # Create sliders and widgets
    N_slider = IntSlider(value=30, min=1, max=100, description=r'Dim($\mathcal{H}$)')
    n_th_a_slider = IntSlider(value=4, min=1, max=10, description=r'$\langle \hat{n}_{E}\rangle$')
    psi0_val_slider = IntSlider(value=9, min=0, max=30, description=r'$\rho_0$')
    kappa_slider = FloatSlider(value=0.01, min=0.0, max=0.5, step=0.01, description=r'$\gamma$')
    Omega_slider = FloatSlider(value=0.01, min=0.0, max=1.0, step=0.01, description=r'$\Omega$')
    k_slider = FloatSlider(value=0.05, min=0.0, max=0.2, step=0.01, description=r'$\kappa$')
    # Create checkboxes for driving and Kerr effects
    driving_checkbox = Checkbox(value=False, description="Include Driving")
    kerr_checkbox = Checkbox(value=False, description="Include Kerr")

    # Create the submit button
    submit_button = Button(description='Run Simulation')

    # Function to simulate and update plot
    def simulate_on_submit(N=30, n_th_a=4, psi0_val=9, kappa=0.03, driving=False, kerr=False, Omega=0.0, k=0.0):
        # Create new figure and axes
        fig, ax = plt.subplots(figsize=(10, 8))
        a = destroy(N)
        # Base Hamiltonian
        H0 = a.dag() * a
        H1 = Omega * (a.dag() + a)
        H2 = k * (a.dag() * a * a.dag() * a)
        psi0 = basis(N, psi0_val)
        # Collapse operators
        c_op_list = []
        rate = kappa * (1 + n_th_a)
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * a)  # Decay operators
        rate = kappa * n_th_a
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * a.dag())  # Excitation operators
        opts = {'store_states': True}
        H = H0
        if driving:
            H = H0 + H1
        if kerr:
            H = H0 + (H1 if driving else 0) + H2
        #         # Solve and create the animation for Fock distribution
        #         medata = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a], options=opts)
        #         W = []
        #         for i in range(len(medata.states)):
        #             W.append(wigner(medata.states[i], xvec, xvec))

        title = f"$\\langle \\hat{{n}}_E\\rangle = {n_th_a}$, $\\gamma = {kappa}$"
        if driving:
            title += f", $\\Omega = {Omega}$"
        if kerr:
            title += f", $\\kappa = {k}$"
        fig.suptitle(title)

        #         # Initialize the image
        #         im = ax.imshow(W[0], extent=[xvec.min(), xvec.max(), xvec.min(), xvec.max()],
        #                        cmap='seismic', vmin=-1.0, vmax=1.0, origin='lower')

        #         # Add colorbar
        #         divider = make_axes_locatable(ax)
        #         cax = divider.append_axes("right", size="5%", pad=0.05)
        #         colorbar = fig.colorbar(im, cax=cax)
        #         colorbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        #         colorbar.set_ticklabels(['-1.00', '-0.50', '0.00', '0.50', '1.00'])

        medata = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a], options=opts)
        W = [wigner(state, xvec, xvec) for state in medata.states]

        max_abs = max(max(abs(np.min(W_frame)), abs(np.max(W_frame))) for W_frame in W)
        vmin, vmax = -max_abs, max_abs

        # Initialize plot with symmetric color limits
        im = ax.imshow(W[0], extent=[xvec.min(), xvec.max(), xvec.min(), xvec.max()],
                       cmap='seismic', vmin=vmin, vmax=vmax, origin='lower',  interpolation='spline16')

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        colorbar = fig.colorbar(im, cax=cax)
        colorbar.set_label(r'$W(\alpha)$')

        # Add axis labels
        ax.set_xlabel(r'Re($\alpha$)')
        ax.set_ylabel(r'Im($\alpha$)')

        def update(n):
            im.set_data(W[n])
            return [im]

        anim = animation.FuncAnimation(fig, update, frames=len(medata.states), blit=True, interval=100)

        plt.close(fig)  # Close the figure to prevent duplicate display

        # Display the animation
        html_anim = HTML(anim.to_jshtml())
        display(html_anim)

    # Function to be called when the button is clicked
    def on_submit(b):
        submit_button.disabled = True  # Disable the submit button
        with output:
            clear_output(wait=True)
            simulate_on_submit(N_slider.value, n_th_a_slider.value, psi0_val_slider.value,
                               kappa_slider.value, driving_checkbox.value, kerr_checkbox.value,
                               Omega_slider.value, k_slider.value)
        submit_button.disabled = False  # Re-enable the submit button

    # Attach the function to the button
    submit_button.on_click(on_submit)

    # Arrange widgets in a box and display controls only once
    controls_box = VBox([HBox([N_slider, n_th_a_slider]),
                         HBox([psi0_val_slider, kappa_slider]),
                         HBox([driving_checkbox, Omega_slider]),
                         HBox([kerr_checkbox, k_slider]),
                         submit_button])
    display(controls_box)
    display(output)

    # Call the simulation function once to show the initial plot
    with output:
        simulate_on_submit(N_slider.value, n_th_a_slider.value, psi0_val_slider.value,
                           kappa_slider.value, driving_checkbox.value, kerr_checkbox.value,
                           Omega_slider.value, k_slider.value)

def Plot_Wigner_3D(tlist=np.linspace(0, 10, 50), xvec=np.linspace(-5, 5, 100)):
    # Create output widget
    output = Output()

    # Create sliders and widgets
    N_slider = IntSlider(value=30, min=1, max=100, description=r'Dim($\mathcal{H}$)')
    n_th_a_slider = IntSlider(value=4, min=1, max=10, description=r'$\langle \hat{n}_{E}\rangle$')
    psi0_val_slider = IntSlider(value=9, min=0, max=30, description=r'$\rho_0$')
    kappa_slider = FloatSlider(value=0.03, min=0.0, max=0.5, step=0.01, description=r'$\gamma$')
    Omega_slider = FloatSlider(value=0.0, min=0.0, max=0.5, step=0.01, description=r'$\Omega$')
    k_slider = FloatSlider(value=0.0, min=0.0, max=0.2, step=0.01, description=r'$\kappa$')
    driving_checkbox = Checkbox(value=False, description="Include Driving")
    kerr_checkbox = Checkbox(value=False, description="Include Kerr")

    # Create the submit button
    submit_button = Button(description='Run Simulation')

    # Function to simulate and update plot
    def simulate_on_submit(N=30, n_th_a=4, psi0_val=9, kappa=0.03, driving=False, kerr=False, Omega=0.0, k=0.0):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        a = destroy(N)
        # Base Hamiltonian
        H0 = a.dag() * a
        H1 = Omega * (a.dag() + a)
        H2 = k * (a.dag() * a * a.dag() * a)
        psi0 = basis(N, psi0_val)
        
        # Collapse operators
        c_op_list = []
        rate = kappa * (1 + n_th_a)
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * a)
        rate = kappa * n_th_a
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * a.dag())
        opts = {'store_states': True}
        H = H0 + (H1 if driving else 0) + (H2 if kerr else 0)

        # Solve and create the animation for Wigner function
        medata = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a], options=opts)
        W = [wigner(state, xvec, xvec) for state in medata.states]
        W = np.array(W)

        # Calculate the symmetric colormap limits based on max absolute value in all frames
        max_abs = max(max(abs(np.min(W_frame)), abs(np.max(W_frame))) for W_frame in W)
        color_vmin, color_vmax = -max_abs, max_abs

        # Start with overall min and max values for the z-axis
        zmin, zmax = W.min(), W.max()

        # Set up meshgrid and labels
        X, Y = np.meshgrid(xvec, xvec)
        ax.set_xlabel(r'Re($\alpha$)')
        ax.set_ylabel(r'Im($\alpha$)')
        ax.set_zlabel(r'W($\alpha)$')

        # Initialize the surface plot and set initial z-limits
        surf = [ax.plot_surface(X, Y, W[0], cmap='seismic', vmin=color_vmin, vmax=color_vmax, linewidth=0, antialiased=False, alpha=0.9)]
        ax.set_zlim(zmin, zmax)

        # Define decay factor for gradual adjustment
        decay_factor = 0.1

        # Update function for the surface plot
        def update(n):
            nonlocal zmin, zmax
            surf[0].remove()

            # Calculate current frame's min and max
            current_zmin, current_zmax = W[n].min(), W[n].max()

            # Gradually adjust zmin and zmax towards the current frame's values
            zmin = decay_factor * current_zmin + (1 - decay_factor) * zmin
            zmax = decay_factor * current_zmax + (1 - decay_factor) * zmax
            ax.set_zlim(zmin, zmax)

            surf[0] = ax.plot_surface(X, Y, W[n], cmap='seismic', vmin=color_vmin, vmax=color_vmax, linewidth=0, antialiased=False, alpha=0.9)
            return surf

        anim = animation.FuncAnimation(fig, update, frames=len(W), blit=False, interval=100)
        plt.close(fig)

        # Display the animation
        html_anim = HTML(anim.to_jshtml())
        display(html_anim)

    # Function to be called when the button is clicked
    def on_submit(b):
        submit_button.disabled = True  # Disable the submit button
        with output:
            clear_output(wait=True)
            simulate_on_submit(N_slider.value, n_th_a_slider.value, psi0_val_slider.value,
                               kappa_slider.value, driving_checkbox.value, kerr_checkbox.value,
                               Omega_slider.value, k_slider.value)
        submit_button.disabled = False  # Re-enable the submit button

    # Attach the function to the button
    submit_button.on_click(on_submit)

    # Arrange widgets in a box and display controls
    controls_box = VBox([HBox([N_slider, n_th_a_slider]),
                         HBox([psi0_val_slider, kappa_slider]),
                         HBox([driving_checkbox, Omega_slider]),
                         HBox([kerr_checkbox, k_slider]),
                         submit_button])
    display(controls_box)
    display(output)

    # Call the simulation function once to show the initial plot
    with output:
        simulate_on_submit(N_slider.value, n_th_a_slider.value, psi0_val_slider.value,
                           kappa_slider.value, driving_checkbox.value, kerr_checkbox.value,
                           Omega_slider.value, k_slider.value)

import matplotlib.colors as colors


def Plot_Q(tlist=np.linspace(0, 100, 100), xvec=np.linspace(-5, 5, 200)):
    # Create output widget
    output = Output()

    # Create sliders and widgets
    N_slider = IntSlider(value=30, min=1, max=100, description=r'Dim($\mathcal{H}$)')
    n_th_a_slider = IntSlider(value=4, min=1, max=10, description=r'$\langle \hat{n}_{E}\rangle$')
    psi0_val_slider = IntSlider(value=9, min=0, max=30, description=r'$\rho_0$')
    kappa_slider = FloatSlider(value=0.03, min=0.0, max=0.5, step=0.01, description=r'$\gamma$')
    Omega_slider = FloatSlider(value=0.0, min=0.0, max=0.5, step=0.01, description=r'$\Omega$')
    k_slider = FloatSlider(value=0.0, min=0.0, max=0.2, step=0.01, description=r'$\kappa$')
    # Create checkboxes for driving and Kerr effects
    driving_checkbox = Checkbox(value=False, description="Include Driving")
    kerr_checkbox = Checkbox(value=False, description="Include Kerr")

    # Create the submit button
    submit_button = Button(description='Run Simulation')

    # Function to simulate and update plot
    def simulate_on_submit(N=30, n_th_a=4, psi0_val=9, kappa=0.03,
                           driving=False, kerr=False, Omega=0.0, k=0.0):
        # Create new figure and axes
        fig, ax = plt.subplots(figsize=(10, 8))
        a = destroy(N)
        # Base Hamiltonian
        H0 = a.dag() * a
        H1 = Omega * (a.dag() + a)
        H2 = k * (a.dag() * a * a.dag() * a)
        psi0 = basis(N, psi0_val)
        # Collapse operators
        c_op_list = []
        rate = kappa * (1 + n_th_a)
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * a)  # Decay operators
        rate = kappa * n_th_a
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * a.dag())  # Excitation operators
        opts = {'store_states': True}
        H = H0
        if driving:
            H += H1
        if kerr:
            H += H2
        # Solve and create the animation for Q-function
        medata = mesolve(H, psi0, tlist, c_op_list, options=opts)
        Q = []
        for state in medata.states:
            Q.append(qfunc(state, xvec, xvec))
        Q = np.array(Q)

        title = f"$\\langle \\hat{{n}}_E\\rangle = {n_th_a}$, $\\gamma = {kappa}$"
        if driving:
            title += f", $\\Omega = {Omega}$"
        if kerr:
            title += f", $\\kappa = {k}$"
        fig.suptitle(title)
        
        def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
            new_cmap = colors.LinearSegmentedColormap.from_list(
                'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                cmap(np.linspace(minval, maxval, n)))
            return new_cmap
        cmap = plt.get_cmap('ocean_r')
        new_cmap = truncate_colormap(cmap, 0.0, 0.6)
        
        # Initialize the image
        im = ax.imshow(Q[0], extent=[xvec.min(), xvec.max(), xvec.min(), xvec.max()],
                       cmap=new_cmap, origin='lower',  interpolation='spline16')

        # Add axis labels
        ax.set_xlabel(r'Re($\alpha$)')
        ax.set_ylabel(r'Im($\alpha$)')

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        colorbar = fig.colorbar(im, cax=cax)
        colorbar.set_label('Q-function Value')

        def update(n):
            im.set_data(Q[n])
            return [im]

        anim = animation.FuncAnimation(fig, update, frames=len(Q), blit=True, interval=100)

        plt.close(fig)  # Close the figure to prevent duplicate display

        # Display the animation
        html_anim = HTML(anim.to_jshtml())
        display(html_anim)

    # Function to be called when the button is clicked
    def on_submit(b):
        submit_button.disabled = True  # Disable the submit button
        with output:
            clear_output(wait=True)
            simulate_on_submit(N_slider.value, n_th_a_slider.value, psi0_val_slider.value,
                               kappa_slider.value, driving_checkbox.value, kerr_checkbox.value,
                               Omega_slider.value, k_slider.value)
        submit_button.disabled = False  # Re-enable the submit button

    # Attach the function to the button
    submit_button.on_click(on_submit)

    # Arrange widgets in a box and display controls only once
    controls_box = VBox([HBox([N_slider, n_th_a_slider]),
                         HBox([psi0_val_slider, kappa_slider]),
                         HBox([driving_checkbox, Omega_slider]),
                         HBox([kerr_checkbox, k_slider]),
                         submit_button])
    display(controls_box)
    display(output)

    # Call the simulation function once to show the initial plot
    with output:
        simulate_on_submit(N_slider.value, n_th_a_slider.value, psi0_val_slider.value,
                           kappa_slider.value, driving_checkbox.value, kerr_checkbox.value,
                           Omega_slider.value, k_slider.value)

def Plot_Q_3D(tlist=np.linspace(0, 10, 50), xvec=np.linspace(-5, 5, 100)):
    # Create output widget
    output = Output()

    # Create sliders and widgets
    N_slider = IntSlider(value=30, min=1, max=100, description=r'Dim($\mathcal{H}$)')
    n_th_a_slider = IntSlider(value=4, min=1, max=10, description=r'$\langle \hat{n}_{E}\rangle$')
    psi0_val_slider = IntSlider(value=9, min=0, max=30, description=r'$\rho_0$')
    kappa_slider = FloatSlider(value=0.03, min=0.0, max=0.5, step=0.01, description=r'$\gamma$')
    Omega_slider = FloatSlider(value=0.0, min=0.0, max=0.5, step=0.01, description=r'$\Omega$')
    k_slider = FloatSlider(value=0.0, min=0.0, max=0.2, step=0.01, description=r'$\kappa$')
    # Create checkboxes for driving and Kerr effects
    driving_checkbox = Checkbox(value=False, description="Include Driving")
    kerr_checkbox = Checkbox(value=False, description="Include Kerr")

    # Create the submit button
    submit_button = Button(description='Run Simulation')

    # Function to simulate and update plot
    def simulate_on_submit(N=30, n_th_a=4, psi0_val=9, kappa=0.03,
                           driving=False, kerr=False, Omega=0.0, k=0.0):
        # Create new figure and axes
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        a = destroy(N)
        # Base Hamiltonian
        H0 = a.dag() * a
        H1 = Omega * (a.dag() + a)
        H2 = k * (a.dag() * a * a.dag() * a)
        psi0 = basis(N, psi0_val)
        # Collapse operators
        c_op_list = []
        rate = kappa * (1 + n_th_a)
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * a)  # Decay operators
        rate = kappa * n_th_a
        if rate > 0.0:
            c_op_list.append(np.sqrt(rate) * a.dag())  # Excitation operators
        opts = {'store_states': True}
        H = H0
        if driving:
            H += H1
        if kerr:
            H += H2
        # Solve and create the animation for Q-function
        medata = mesolve(H, psi0, tlist, c_op_list, options=opts)
        Q = []
        for state in medata.states:
            Q.append(qfunc(state, xvec, xvec))
        Q = np.array(Q)
        max_abs = max(max(abs(np.min(Q_frame)), abs(np.max(Q_frame))) for Q_frame in Q)
        color_vmin, color_vmax = -max_abs, max_abs
        
        title = f"$\\langle \\hat{{n}}_E\\rangle = {n_th_a}$, $\\gamma = {kappa}$"
        if driving:
            title += f", $\\Omega = {Omega}$"
        if kerr:
            title += f", $\\kappa = {k}$"
        ax.set_title(title)

        # Create meshgrid for plotting
        X, Y = np.meshgrid(xvec, xvec)

        # Set axis labels
        ax.set_xlabel(r'Re($\alpha$)')
        ax.set_ylabel(r'Im($\alpha$)')
        ax.set_zlabel('Q-function Value')

        # Set axis limits
        ax.set_xlim(xvec.min(), xvec.max())
        ax.set_ylim(xvec.min(), xvec.max())
        ax.set_zlim(0, Q.max())
        
        def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
            new_cmap = colors.LinearSegmentedColormap.from_list(
                'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
                cmap(np.linspace(minval, maxval, n)))
            return new_cmap
        cmap = plt.get_cmap('ocean_r')
        new_cmap = truncate_colormap(cmap, 0.0, 0.6)
        
        
        
        # Initialize the surface plot and store it in a list
        surf = [ax.plot_surface(X, Y, Q[0], cmap=new_cmap, linewidth=0, antialiased=False, alpha=0.9)]

        # Function to update the surface plot
        def update(n):
            surf[0].remove()  # Remove the previous surface
            surf[0] = ax.plot_surface(X, Y, Q[n],cmap=new_cmap, linewidth=0, antialiased=False, alpha=0.9)
            return surf

        anim = animation.FuncAnimation(fig, update, frames=len(Q), blit=False, interval=100)

        plt.close(fig)  # Close the figure to prevent duplicate display

        # Display the animation
        html_anim = HTML(anim.to_jshtml())
        display(html_anim)

    # Function to be called when the button is clicked
    def on_submit(b):
        submit_button.disabled = True  # Disable the submit button
        with output:
            clear_output(wait=True)
            simulate_on_submit(N_slider.value, n_th_a_slider.value, psi0_val_slider.value,
                               kappa_slider.value, driving_checkbox.value, kerr_checkbox.value,
                               Omega_slider.value, k_slider.value)
        submit_button.disabled = False  # Re-enable the submit button

    # Attach the function to the button
    submit_button.on_click(on_submit)

    # Arrange widgets in a box and display controls only once
    controls_box = VBox([HBox([N_slider, n_th_a_slider]),
                         HBox([psi0_val_slider, kappa_slider]),
                         HBox([driving_checkbox, Omega_slider]),
                         HBox([kerr_checkbox, k_slider]),
                         submit_button])
    display(controls_box)
    display(output)

    # Call the simulation function once to show the initial plot
    with output:
        simulate_on_submit(N_slider.value, n_th_a_slider.value, psi0_val_slider.value,
                           kappa_slider.value, driving_checkbox.value, kerr_checkbox.value,
                           Omega_slider.value, k_slider.value)

def Examine_Steady_State():
    # Create output widget
    output = Output()

    # Create sliders and widgets
    N_slider = IntSlider(value=30, min=5, max=100, description=r'Dim($\mathcal{H}$)', continuous_update=True)
    n_th_slider = IntSlider(value=5, min=0, max=10, description=r'$\langle n_{th} \rangle$', continuous_update=True)
    Omega_slider = FloatSlider(value=0.0, min=0.0, max=1.0, step=0.01, description=r'$\Omega$', continuous_update=True)
    kappa_slider = FloatSlider(value=0.1, min=0.01, max=1.0, step=0.01, description=r'$\gamma$', continuous_update=True)
    
    # Create checkboxes to control the display of thermal and coherent state plots
    show_thermal_checkbox = Checkbox(value=True, description="Show Thermal State")
    show_coherent_checkbox = Checkbox(value=True, description="Show Coherent State")

    # Function to simulate and update plot
    def simulate_on_update(*args):
        with output:
            clear_output(wait=True)
            # Operators
            a = destroy(N_slider.value)

            # Hamiltonian: H = Omega * (a + a.dag())
            H = Omega_slider.value * (a + a.dag())

            # Collapse operators for damping and thermal noise
            c_ops = []
            if kappa_slider.value > 0.0:
                c_ops.append(np.sqrt(kappa_slider.value * (1 + n_th_slider.value)) * a)
                if n_th_slider.value > 0:
                    c_ops.append(np.sqrt(kappa_slider.value * n_th_slider.value) * a.dag())

            # Compute the steady-state density matrix
            rho_ss = steadystate(H, c_ops)

            # Number state distribution
            n = np.arange(N_slider.value)
            prob_n = np.real(np.diag(rho_ss.full()))

            # Number operator and variance
            n_op = a.dag() * a
            n_mean_ss = expect(n_op, rho_ss)
            delta_N_ss = np.sqrt(variance(n_op, rho_ss))

            # For thermal state
            n_th = n_th_slider.value
            delta_N_th = np.sqrt(n_th ** 2 + n_th)

            # For coherent state
            delta_N_coh = np.sqrt(n_mean_ss)

            # Quadrature operators
            x = (a + a.dag()) / np.sqrt(2)
            p = (a - a.dag()) / (1j * np.sqrt(2))
            delta_x_ss = np.sqrt(variance(x, rho_ss))
            delta_p_ss = np.sqrt(variance(p, rho_ss))

            # Theoretical quadrature variances
            if Omega_slider.value == 0.0:
                # Thermal state quadrature variances
                delta_x_th = delta_p_th = np.sqrt((2 * n_th + 1) / 2)
            else:
                # Coherent state quadrature variances
                delta_x_th = delta_p_th = 1 / np.sqrt(2)
            product_var_th = delta_x_th * delta_p_th

            # Theoretical number operator variances for coherent state remains sqrt(n_mean_ss)

            # Prepare theoretical states for plotting
            rho_th = thermal_dm(N_slider.value, n_th_slider.value)
            prob_n_th = np.real(np.diag(rho_th.full()))

            alpha = np.sqrt(n_mean_ss)
            rho_coh = coherent_dm(N_slider.value, alpha)
            prob_n_coh = np.real(np.diag(rho_coh.full()))

            # Plotting Fock state distributions
            fig, ax = plt.subplots(1, 1, figsize=(10, 5))
            width = 0.25
            ax.bar(n, prob_n, width=width, color='skyblue', edgecolor='black', label='Steady State')

            if show_thermal_checkbox.value:
                ax.bar(n - width, prob_n_th, width=width, color='red', edgecolor='black', alpha=0.5, label='Thermal State')

            if show_coherent_checkbox.value:
                ax.bar(n + width, prob_n_coh, width=width, color='green', edgecolor='black', alpha=0.5, label='Coherent State')

            ax.set_xlabel('Fock state $n$')
            ax.set_ylabel('Occupation probability $P(n)$')
            ax.set_title(r'Fock Distributions $\langle n_{th} \rangle= $' + '{}'.format(n_th_slider.value) + r', $\Omega = $' + '{}'.format(Omega_slider.value))
            ax.set_xlim(-0.5, N_slider.value - 0.5)
            ax.legend()
            plt.show()

            # Print number operator statistics
            print(f"Number operator statistics:")
            print(f"Mean photon number ⟨N⟩ = {n_mean_ss:.4f}")
            print(f"Variance ΔN = {delta_N_ss:.4f}")

            # Print quadrature variances
            print("\nQuadrature variances:")
            print(f"Δx = {delta_x_ss:.4f}")
            print(f"Δp = {delta_p_ss:.4f}")
            print(f"Product Δx * Δp = {delta_x_ss * delta_p_ss:.4f}")
            print(f"Heisenberg uncertainty limit: 0.5")

            # Print theoretical predictions
            print("\nTheoretical predictions:")
            print(f"Expected for thermal state with ⟨n_th⟩ = {n_th_slider.value}:")
            print(f"    Variance ΔN_th = sqrt(n_th^2 + n_th) = {delta_N_th:.4f}")
            print(f"    Quadrature Variances Δx_th = Δp_th = sqrt((2 * n_th + 1)/2) = {delta_x_th:.4f}")
            print("Expected for coherent state:")
            print(f"    Variance ΔN_coh = sqrt(⟨N⟩) = {delta_N_coh:.4f}")
            print(f"    Quadrature Variances Δx_coh = Δp_coh = 1/sqrt(2) ≈ {1/np.sqrt(2):.4f}")
            print("      (Where ⟨N⟩ is computed for the steady state)")

            # Conclusion based on number operator variances
            print("\nConclusion based on Number Operator Variance:")
            if abs(delta_N_ss - delta_N_th) < abs(delta_N_ss - delta_N_coh):
                print("The steady state is closer to a thermal state since the number state variance is closer to ΔN_th.")
            else:
                print("The steady state is closer to a coherent state since the number state variance is closer to ΔN_coh.")

            # Conclusion based on quadrature variances
            print("\nConclusion based on Quadrature Variances:")
            uncertainty_product_ss = delta_x_ss * delta_p_ss
            if uncertainty_product_ss > 0.5 + 1e-3:  # Adding a small tolerance
                print("The steady state exceeds the Heisenberg uncertainty limit, indicating a mixed or thermal-like state.")
            elif np.isclose(uncertainty_product_ss, 0.5, atol=1e-3):
                print("The steady state saturates the Heisenberg uncertainty limit, indicating a coherent or minimum uncertainty state.")
            else:
                print("The steady state has uncertainty below the Heisenberg limit, which is not physically possible for standard states.")

    # Attach the function to be called on slider/checkbox changes
    N_slider.observe(simulate_on_update, names='value')
    n_th_slider.observe(simulate_on_update, names='value')
    Omega_slider.observe(simulate_on_update, names='value')
    kappa_slider.observe(simulate_on_update, names='value')
    show_thermal_checkbox.observe(simulate_on_update, names='value')
    show_coherent_checkbox.observe(simulate_on_update, names='value')

    # Arrange widgets in a box and display controls
    controls_box = VBox([
        HBox([N_slider, n_th_slider]),
        HBox([Omega_slider, kappa_slider]),
        HBox([show_thermal_checkbox, show_coherent_checkbox]),
    ])
    display(controls_box)
    display(output)

    # Initial plot
    simulate_on_update()



def Demonstrate_NonClassicality():
    # Create output widget
    output = Output()

    # Create sliders and widgets
    N_slider = IntSlider(value=30, min=5, max=120, step=1, description=r'Dim($\mathcal{H}$)', continuous_update=False)
    n_th_slider = IntSlider(value=5, min=1, max=15, step=1, description=r'$\langle n_{th} \rangle$', continuous_update=False)
    Omega_slider = FloatSlider(value=0.2, min=0.0, max=1.0, step=0.01, description=r'$\Omega$', continuous_update=False)
    kappa_slider = FloatSlider(value=0.1, min=0.01, max=1.0, step=0.01, description=r'$\gamma$', continuous_update=False)
    kerr_slider = FloatSlider(value=0.01, min=0.0, max=0.1, step=0.001, description=r'$\kappa$', continuous_update=False)
    xi_slider = FloatSlider(value=0.0, min=-1.0, max=1.0, step=0.01, description=r'$\xi$', continuous_update=False)

    # Single slider for xvec range
    xvec_range_slider = IntSlider(value=5, min=2, max=20, step=1, description='Domain Size', continuous_update=False)
    xvec_points_slider = IntSlider(value=200, min=50, max=500, step=50, description='xvec points', continuous_update=False)

    # Time parameters
    tlist = np.linspace(0, 10, 100)

    # Create checkboxes
    show_wigner_checkbox = Checkbox(value=True, description="Show Wigner Function")
    show_mandelQ_checkbox = Checkbox(value=True, description="Calculate Mandel Q")
    show_squeezing_checkbox = Checkbox(value=True, description="Check Squeezing")

    # Create the submit button
    submit_button = Button(description='Run Simulation')

    # Function to simulate and update results
    def simulate_on_submit(b):
        submit_button.disabled = True  # Disable the submit button
        with output:
            clear_output(wait=True)
            N = int(N_slider.value)  # Ensure N is integer
            n_th = n_th_slider.value
            Omega = Omega_slider.value
            kappa = kappa_slider.value
            kerr = kerr_slider.value
            xi = xi_slider.value
            xvec_max = xvec_range_slider.value
            xvec_points = xvec_points_slider.value

            # Operators
            a = destroy(N)

            # Hamiltonian
            H = a*a.dag()+ Omega * (a + a.dag()) + kerr *(a.dag()*a* a.dag()*a) + xi*(a.dag()*a.dag() + a*a)

            # Collapse operators
            c_ops = []
            # Decay due to coupling with the environment
            if kappa > 0.0:
                c_ops.append(np.sqrt(kappa * (1 + n_th)) * a)
                if n_th > 0:
                    c_ops.append(np.sqrt(kappa * n_th) * a.dag())

            # Initial state (vacuum)
            psi0 = basis(N, 0)

            # Solve the master equation
            result = mesolve(H, psi0, tlist, c_ops, [])

            # Steady-state density matrix
            rho_ss = steadystate(H, c_ops)

            
            if show_wigner_checkbox.value:
                xvec = np.linspace(-xvec_max, xvec_max, xvec_points)
                W_ss = wigner(rho_ss, xvec, xvec)
                X, Y = np.meshgrid(xvec, xvec)

                # Calculate min and max of W_ss
                W_min = W_ss.min()
                W_max = W_ss.max()
                vmax = np.abs(W_max)
                vmin = -np.abs(W_max)
                if abs(W_min) > abs(W_max):
                    vmax = np.abs(W_min)
                    vmin = -np.abs(W_min)
                # Plot Wigner function with custom levels
                fig, ax = plt.subplots(1, 1, figsize=(6, 5))
                im = ax.imshow(W_ss, extent=[xvec.min(), xvec.max(), xvec.min(), xvec.max()],
                               cmap='seismic', vmin=vmin, vmax=vmax, origin='lower',  interpolation='spline16')
                # Add axis labels
                ax.set_xlabel(r'Re($\alpha$)')
                ax.set_ylabel(r'Im($\alpha$)')

                ax.set_title('Wigner Function of Steady State')
                plt.show()

                # Check for negative values in Wigner function
                if W_min < 0:
                    print(f"Minimum Wigner function value: {W_min:.4e}")
                    print("The Wigner function has negative regions: State is non-classical.")
                else:
                    print("The Wigner function is non-negative: State is classical.")

            # Compute Mandel Q parameter
            if show_mandelQ_checkbox.value:
                n_op = a.dag() * a
                n_mean = expect(n_op, rho_ss)
                n_var = expect(n_op ** 2, rho_ss) - n_mean ** 2
                Q = (n_var - n_mean) / n_mean if n_mean != 0 else np.inf
                print(f"\nMandel Q parameter: Q = {Q:.4f}")
                if Q < 0:
                    print("Negative Mandel Q parameter: State may be non-classical (sub-Poissonian).")
                else:
                    print("Mandel Q parameter is zero or positive: State is classical.")

            # Check for quadrature squeezing
            if show_squeezing_checkbox.value:
                x = (a + a.dag()) / np.sqrt(2)
                p = -1j * (a - a.dag()) / np.sqrt(2)
                delta_x = np.sqrt(variance(x, rho_ss))
                delta_p = np.sqrt(variance(p, rho_ss))
                print(f"\nQuadrature variances:")
                print(f"Δx = {delta_x:.4f}, Δp = {delta_p:.4f}")
                if delta_x < 1/np.sqrt(2) or delta_p < 1/np.sqrt(2):
                    print("One of the quadrature variances indicate state is squeezed (non-classical).")
                else:
                    print("Quadrature variances indicate state is not squeezed.")
        submit_button.disabled = False  # Re-enable the submit button

    # Attach the function to the button
    submit_button.on_click(simulate_on_submit)

    # Arrange widgets in a box and display controls
    controls_box = VBox([
        HBox([N_slider, n_th_slider]),
        HBox([Omega_slider, kappa_slider, kerr_slider, xi_slider]),
        HBox([xvec_range_slider, xvec_points_slider]),
        HBox([show_wigner_checkbox, show_mandelQ_checkbox, show_squeezing_checkbox]),
        submit_button
    ])
    display(controls_box)
    display(output)

    # Call the simulation function once to show the initial results
    simulate_on_submit(None)

