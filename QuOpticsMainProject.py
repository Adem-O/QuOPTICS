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
    N_slider = IntSlider(value=30, min=1, max=100, description=r'Dim(H)')
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
    N_slider = IntSlider(value=30, min=1, max=100, description=r'Dim(H)')
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
        bars = ax.bar(n, probs[0], align='center')
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
    N_slider = IntSlider(value=30, min=1, max=100, description=r'Dim(H)')
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
            H = H0 + H1
        if kerr:
            H = H0 + (H1 if driving else 0) + H2
        # Solve and create the animation for Fock distribution
        medata = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a], options=opts)
        W = []
        for i in range(len(medata.states)):
            W.append(wigner(medata.states[i], xvec, xvec))

        title = f"$\\langle \\hat{{n}}_E\\rangle = {n_th_a}$, $\\gamma = {kappa}$"
        if driving:
            title += f", $\\Omega = {Omega}$"
        if kerr:
            title += f", $\\kappa = {k}$"
        fig.suptitle(title)

        # Initialize the image
        im = ax.imshow(W[0], extent=[xvec.min(), xvec.max(), xvec.min(), xvec.max()],
                       cmap='seismic', vmin=-1.0, vmax=1.0, origin='lower')

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        colorbar = fig.colorbar(im, cax=cax)
        colorbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        colorbar.set_ticklabels(['-1.00', '-0.50', '0.00', '0.50', '1.00'])
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
    N_slider = IntSlider(value=30, min=1, max=100, description=r'Dim(H)')
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
    def simulate_on_submit(N=30, n_th_a=4, psi0_val=9, kappa=0.03, driving=False, kerr=False, Omega=0.0, k=0.0):
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
        # Solve and create the animation for Wigner function
        medata = mesolve(H, psi0, tlist, c_op_list, [a.dag() * a], options=opts)
        W = []
        for i in range(len(medata.states)):
            W.append(wigner(medata.states[i], xvec, xvec))
        W = np.array(W)

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
        ax.set_zlabel(r'W($\alpha)$')

        # Set axis limits
        ax.set_xlim(xvec.min(), xvec.max())
        ax.set_ylim(xvec.min(), xvec.max())
        ax.set_zlim(-1.0,1.0)

        # Initialize the surface plot and store it in a list
        surf = [ax.plot_surface(X, Y, W[0], cmap='viridis', linewidth=0, antialiased=False, alpha=0.7)]

        # Function to update the surface plot
        def update(n):
            surf[0].remove()  # Remove the previous surface
            surf[0] = ax.plot_surface(X, Y, W[n], cmap='viridis', linewidth=0, antialiased=False, alpha=0.7)
            return surf

        anim = animation.FuncAnimation(fig, update, frames=len(W), blit=False, interval=100)

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


def Plot_Q(tlist=np.linspace(0, 100, 100), xvec=np.linspace(-5, 5, 200)):
    # Create output widget
    output = Output()

    # Create sliders and widgets
    N_slider = IntSlider(value=30, min=1, max=100, description=r'Dim(H)')
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

        # Initialize the image
        im = ax.imshow(Q[0], extent=[xvec.min(), xvec.max(), xvec.min(), xvec.max()],
                       cmap='viridis', vmin=0, vmax=Q.max(), origin='lower')

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
    N_slider = IntSlider(value=30, min=1, max=100, description=r'Dim(H)')
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

        # Initialize the surface plot and store it in a list
        surf = [ax.plot_surface(X, Y, Q[0], cmap='viridis', linewidth=0, antialiased=False, alpha=0.7)]

        # Function to update the surface plot
        def update(n):
            surf[0].remove()  # Remove the previous surface
            surf[0] = ax.plot_surface(X, Y, Q[n], cmap='viridis', linewidth=0, antialiased=False, alpha=0.7)
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

