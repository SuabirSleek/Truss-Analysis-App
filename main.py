import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Circle, Arrow

class TrussSystem:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.joints = []
        self.members = []
        self.loads = {}
        self.supports = {}
        self.has_horizontal_restraint = False
        self.has_vertical_restraint = False

    def check_stability(self):
        """Check if the truss is stable using the 2j = m + r rule"""
        j = len(self.joints)
        m = len(self.members)
        r = sum(2 if sup == "pinned" else 1 for sup in self.supports.values())
        
        if 2 * j > m + r:
            return False, f"Unstable: 2j ({2*j}) > m + r ({m}+{r}={m+r})"
        elif 2 * j < m + r:
            return False, f"Over-constrained: 2j ({2*j}) < m + r ({m}+{r}={m+r})"
        return True, "Stable: 2j = m + r"

    def analyze(self):
        num_joints = len(self.joints)
        num_members = len(self.members)
        
        # Check stability first
        stable, stability_msg = self.check_stability()
        if not stable:
            return None, None, stability_msg
        
        # Check support validity
        self.has_horizontal_restraint = any(
            sup == "pinned" for sup in self.supports.values()
        )
        self.has_vertical_restraint = len(self.supports) > 0
        
        # Calculate required reaction columns
        num_reactions = 0
        for sup_type in self.supports.values():
            num_reactions += 2 if sup_type == "pinned" else 1
        
        # Initialize matrices
        A = np.zeros((2*num_joints, num_members + num_reactions))
        b = np.zeros(2*num_joints)

        # Member force contributions
        for i, (start, end) in enumerate(self.members):
            x1, y1 = self.joints[start]
            x2, y2 = self.joints[end]
            L = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if L == 0:
                return None, None, "Zero-length member detected"
            cos = (x2-x1)/L
            sin = (y2-y1)/L
            
            A[2*start, i] = cos
            A[2*start+1, i] = sin
            A[2*end, i] = -cos
            A[2*end+1, i] = -sin

        # Support reactions
        react_col = num_members
        for joint, sup_type in self.supports.items():
            if sup_type == "pinned":
                A[2*joint, react_col] = 1    # Rx
                A[2*joint+1, react_col+1] = 1  # Ry
                react_col += 2
            elif sup_type == "roller":
                A[2*joint+1, react_col] = 1  # Ry
                react_col += 1

        # Applied loads
        for joint, (fx, fy) in self.loads.items():
            b[2*joint] = -fx
            b[2*joint+1] = -fy

        try:
            forces = np.linalg.solve(A, b)
            member_forces = forces[:num_members]
            reactions = forces[num_members:]
            return member_forces, reactions, stability_msg
        except np.linalg.LinAlgError as e:
            return None, None, f"Matrix solution error: {str(e)}"

class TrussApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Truss Analysis Tool")
        self.root.geometry("1100x750")
        self.truss = TrussSystem()
        self.setup_ui()
        
    def setup_ui(self):
        # Control Frame
        control_frame = tk.Frame(self.root, padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Plot Frame
        self.plot_frame = tk.Frame(self.root)
        self.plot_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        # Step-by-step guide
        self.step_label = tk.Label(control_frame, text="Step 1: Define Joints", 
                                  font=('Arial', 10, 'bold'))
        self.step_label.pack(fill=tk.X, pady=5)
        
        # Buttons
        button_frame = tk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        self.joints_btn = ttk.Button(button_frame, text="1. Define Joints", command=self.input_joints)
        self.joints_btn.pack(fill=tk.X, pady=2)
        
        self.members_btn = ttk.Button(button_frame, text="2. Define Members", command=self.input_members)
        self.members_btn.pack(fill=tk.X, pady=2)
        
        self.loads_btn = ttk.Button(button_frame, text="3. Apply Loads", command=self.input_loads)
        self.loads_btn.pack(fill=tk.X, pady=2)
        
        self.supports_btn = ttk.Button(button_frame, text="4. Apply Supports", command=self.input_supports)
        self.supports_btn.pack(fill=tk.X, pady=2)
        
        self.analyze_btn = ttk.Button(button_frame, text="5. Analyze", command=self.analyze)
        self.analyze_btn.pack(fill=tk.X, pady=2)
        
        self.clear_btn = ttk.Button(button_frame, text="6. Clear All", command=self.clear)
        self.clear_btn.pack(fill=tk.X, pady=2)
        
        # Results Display
        result_frame = tk.Frame(control_frame)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        scrollbar = tk.Scrollbar(result_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.result_text = tk.Text(result_frame, height=20, width=45, 
                                 yscrollcommand=scrollbar.set)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=self.result_text.yview)
        
        # Initialize plot
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.update_button_states()
    
    def update_button_states(self):
        self.joints_btn.config(state=tk.NORMAL)
        self.members_btn.config(state=tk.NORMAL if len(self.truss.joints) >= 2 else tk.DISABLED)
        self.loads_btn.config(state=tk.NORMAL if len(self.truss.joints) >= 1 else tk.DISABLED)
        self.supports_btn.config(state=tk.NORMAL if len(self.truss.joints) >= 1 else tk.DISABLED)
        self.analyze_btn.config(state=tk.NORMAL if (len(self.truss.joints) >= 2 and 
                                                 len(self.truss.members) >= 1 and 
                                                 len(self.truss.supports) >= 1) else tk.DISABLED)
        
        if len(self.truss.joints) < 2:
            self.step_label.config(text="Step 1: Define at least 2 joints")
        elif len(self.truss.members) < 1:
            self.step_label.config(text="Step 2: Define members")
        elif len(self.truss.supports) < 1:
            self.step_label.config(text="Step 3: Apply supports")
        elif len(self.truss.loads) < 1:
            self.step_label.config(text="Step 4: Apply loads (optional)")
        else:
            self.step_label.config(text="Ready to analyze")
    
    def input_joints(self):
        n = simpledialog.askinteger("Input", "Number of joints:", minvalue=2, initialvalue=3)
        if not n: return
        
        self.truss.reset()
        for i in range(n):
            while True:
                default = f"{i},0" if i == 0 else f"{i*2},0" if i == 1 else f"{i},3"
                coords = simpledialog.askstring("Input", f"Joint {i} coordinates (x,y):", initialvalue=default)
                if not coords and i == 0: continue
                if not coords: break
                try:
                    x, y = map(float, coords.split(','))
                    self.truss.joints.append((x, y))
                    break
                except:
                    messagebox.showerror("Error", "Invalid format! Use 'x,y'")
        self.plot_structure()
        self.update_button_states()
    
    def input_members(self):
        if len(self.truss.joints) < 2:
            messagebox.showerror("Error", "Need at least 2 joints to define members!")
            return
            
        n = simpledialog.askinteger("Input", "Number of members:", minvalue=1, initialvalue=len(self.truss.joints))
        if not n: return
        
        self.truss.members = []
        for i in range(n):
            while True:
                default = f"{i},{i+1}" if i < len(self.truss.joints)-1 else "0,1"
                members = simpledialog.askstring("Input", f"Member {i} joints (start,end):", initialvalue=default)
                if not members and i == 0: continue
                if not members: break
                try:
                    start, end = map(int, members.split(','))
                    if start == end:
                        messagebox.showerror("Error", "Start and end cannot be the same!")
                        continue
                    if (0 <= start < len(self.truss.joints) and 0 <= end < len(self.truss.joints)):
                        if (start, end) in self.truss.members or (end, start) in self.truss.members:
                            messagebox.showerror("Error", "This member already exists!")
                            continue
                        self.truss.members.append((start, end))
                        break
                    else:
                        messagebox.showerror("Error", f"Invalid joint indices! Must be 0-{len(self.truss.joints)-1}")
                except:
                    messagebox.showerror("Error", "Invalid format! Use 'start,end'")
        self.plot_structure()
        self.update_button_states()
    
    def input_loads(self):
        if not self.truss.joints:
            messagebox.showerror("Error", "Define joints first!")
            return
            
        self.truss.loads = {}
        for i in range(len(self.truss.joints)):
            load = simpledialog.askstring("Input", f"Load at joint {i} (fx,fy in kN):", initialvalue="0,0")
            if load:
                try:
                    fx, fy = map(float, load.split(','))
                    if fx != 0 or fy != 0:
                        self.truss.loads[i] = (fx, fy)
                except:
                    messagebox.showerror("Error", "Invalid format! Use 'fx,fy'")
        self.plot_structure()
        self.update_button_states()
    
    def input_supports(self):
        if not self.truss.joints:
            messagebox.showerror("Error", "Define joints first!")
            return
            
        self.truss.supports = {}
        for i in range(len(self.truss.joints)):
            sup = simpledialog.askstring("Input", 
                f"Support at joint {i}:\n"
                "'pinned' (fixes X & Y) or 'roller' (fixes Y only),\n"
                "or leave empty for no support:",
                initialvalue="pinned" if i == 0 else "roller" if i == 1 else "")
            if sup:
                if sup.lower() in ["pinned", "roller"]:
                    self.truss.supports[i] = sup.lower()
                else:
                    messagebox.showerror("Error", "Must be 'pinned' or 'roller'")
        self.plot_structure()
        self.update_button_states()
    
    def plot_structure(self):
        self.ax.clear()
        
        # Plot joints
        if self.truss.joints:
            x, y = zip(*self.truss.joints)
            self.ax.scatter(x, y, color='blue', s=100, zorder=5)
            for i, (xi, yi) in enumerate(self.truss.joints):
                self.ax.text(xi, yi, str(i), ha='center', va='center', color='white', zorder=6)
        
        # Plot members
        for start, end in self.truss.members:
            x1, y1 = self.truss.joints[start]
            x2, y2 = self.truss.joints[end]
            self.ax.plot([x1, x2], [y1, y2], 'k-', linewidth=2, zorder=1)
        
        # Plot loads
        scale = self.get_force_scale()
        for joint, (fx, fy) in self.truss.loads.items():
            x, y = self.truss.joints[joint]
            if fx != 0 or fy != 0:
                force_mag = np.sqrt(fx**2 + fy**2)
                if force_mag > 0:
                    dx, dy = fx/force_mag * scale, fy/force_mag * scale
                    arrow = Arrow(x, y, dx, dy, width=scale/2, color='red', zorder=4)
                    self.ax.add_patch(arrow)
                    self.ax.text(x + dx*1.2, y + dy*1.2, f"({fx:.1f}, {fy:.1f}) kN", 
                                bbox=dict(facecolor='white', alpha=0.8), zorder=7)
        
        # Plot supports with improved visualization
        if self.truss.joints:
            x_coords, y_coords = zip(*self.truss.joints)
            max_dim = max(max(x_coords)-min(x_coords), max(y_coords)-min(y_coords)) or 1
            support_size = max(0.3, max_dim / 8)
            
            for joint, sup_type in self.truss.supports.items():
                x, y = self.truss.joints[joint]
                
                if sup_type == "pinned":
                    # Fixed support - triangle with vertical line
                    triangle = Polygon([
                        [x, y-support_size*0.5],
                        [x-support_size*0.4, y-support_size],
                        [x+support_size*0.4, y-support_size]
                    ], color='darkgreen', zorder=3)
                    self.ax.add_patch(triangle)
                    self.ax.plot([x, x], [y-support_size*0.5, y], 
                               color='darkgreen', linewidth=2, zorder=3)
                    
                elif sup_type == "roller":
                    # Roller support - circle with base and rollers
                    circle = Circle((x, y-support_size*0.75), support_size*0.3,
                                  color='darkorange', zorder=3)
                    self.ax.add_patch(circle)
                    # Base
                    self.ax.plot([x-support_size*0.5, x+support_size*0.5],
                               [y-support_size, y-support_size],
                               color='darkorange', linewidth=3, zorder=3)
                    # Rollers
                    for i in range(3):
                        roller_x = x - support_size*0.4 + i*support_size*0.4
                        roller = Circle((roller_x, y-support_size*0.9), support_size*0.1,
                                      color='darkorange', zorder=4)
                        self.ax.add_patch(roller)
        
        # Set plot limits
        if self.truss.joints:
            x_padding = max(2, (max(x_coords) - min(x_coords)) * 0.2)
            y_padding = max(2, (max(y_coords) - min(y_coords)) * 0.2)
            self.ax.set_xlim(min(x_coords) - x_padding, max(x_coords) + x_padding)
            self.ax.set_ylim(min(y_coords) - y_padding, max(y_coords) + y_padding)
        
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.set_title("Truss Diagram")
        self.canvas.draw()
    
    def get_force_scale(self):
        if not self.truss.joints:
            return 1.0
        
        x_coords, y_coords = zip(*self.truss.joints)
        if len(x_coords) < 2:
            return 1.0
            
        max_dim = max(max(x_coords) - min(x_coords), max(y_coords) - min(y_coords))
        return max_dim / 5
    
    def analyze(self):
        if len(self.truss.joints) < 2:
            messagebox.showerror("Error", "Need at least 2 joints!")
            return
            
        if len(self.truss.members) < 1:
            messagebox.showerror("Error", "Need at least 1 member!")
            return
            
        if len(self.truss.supports) < 1:
            messagebox.showerror("Error", "Need at least 1 support!")
            return
            
        member_forces, reactions, stability_msg = self.truss.analyze()
        
        if member_forces is None:
            messagebox.showerror("Analysis Failed", 
                f"Truss analysis failed!\n\n"
                f"Possible causes:\n"
                f"1. Unstable truss configuration\n"
                f"2. Insufficient supports\n"
                f"3. Special geometry causing singularity\n\n"
                f"Details: {stability_msg}")
            return
        
        # Display results
        result = "=== TRUSS ANALYSIS RESULTS ===\n\n"
        result += f"Stability Check: {stability_msg}\n\n"
        
        # Member forces
        result += "=== MEMBER FORCES ===\n"
        max_force = max(abs(f) for f in member_forces) if member_forces.size > 0 else 0
        for i, (force, (start, end)) in enumerate(zip(member_forces, self.truss.members)):
            x1, y1 = self.truss.joints[start]
            x2, y2 = self.truss.joints[end]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            
            force_type = "Tension" if force > 0 else "Compression"
            result += f"Member {i} (J{start}-J{end}): {abs(force):.2f} kN ({force_type}), Length: {length:.2f} m\n"
        
        # Reaction forces
        result += "\n=== REACTION FORCES ===\n"
        react_idx = 0
        for joint, sup_type in self.truss.supports.items():
            if sup_type == "pinned":
                rx = reactions[react_idx] if react_idx < len(reactions) else 0
                ry = reactions[react_idx+1] if (react_idx+1) < len(reactions) else 0
                result += f"Joint {joint} (Pinned): Rx={rx:.2f} kN, Ry={ry:.2f} kN\n"
                react_idx += 2
            elif sup_type == "roller":
                ry = reactions[react_idx] if react_idx < len(reactions) else 0
                result += f"Joint {joint} (Roller): Ry={ry:.2f} kN\n"
                react_idx += 1
        
        # Check equilibrium
        sum_fx = sum(fx for fx, fy in self.truss.loads.values())
        sum_fy = sum(fy for fx, fy in self.truss.loads.values())
        
        sum_rx = 0
        sum_ry = 0
        react_idx = 0
        for sup_type in self.truss.supports.values():
            if sup_type == "pinned":
                sum_rx += reactions[react_idx] if react_idx < len(reactions) else 0
                sum_ry += reactions[react_idx+1] if (react_idx+1) < len(reactions) else 0
                react_idx += 2
            elif sup_type == "roller":
                sum_ry += reactions[react_idx] if react_idx < len(reactions) else 0
                react_idx += 1
        
        result += f"\nSum of applied forces: Fx={sum_fx:.2f} kN, Fy={sum_fy:.2f} kN\n"
        result += f"Sum of reactions: Rx={sum_rx:.2f} kN, Ry={sum_ry:.2f} kN\n"
        
        # Stability warnings
        if not self.truss.has_horizontal_restraint:
            result += "\n⚠️ Warning: No horizontal restraint detected!\n"
            result += "Structure may be unstable under horizontal loads.\n"
            result += "Recommendation: Add at least one pinned support.\n"
        
        if not self.truss.has_vertical_restraint:
            result += "\n⚠️ Warning: No vertical restraint detected!\n"
            result += "Structure may be unstable under vertical loads.\n"
        
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)
        
        # Visualize forces on plot
        self.plot_structure()
        if member_forces.size > 0:
            max_force = max(abs(f) for f in member_forces) or 1
            
            for i, ((start, end), force) in enumerate(zip(self.truss.members, member_forces)):
                x1, y1 = self.truss.joints[start]
                x2, y2 = self.truss.joints[end]
                mid_x = (x1 + x2)/2
                mid_y = (y1 + y2)/2
                
                linewidth = 2 + 3*abs(force)/max_force
                color = 'red' if force < 0 else 'green'
                self.ax.plot([x1, x2], [y1, y2], linewidth=linewidth, color=color, zorder=2)
                
                self.ax.text(mid_x, mid_y, f"{abs(force):.1f} kN", 
                            bbox=dict(facecolor='white', alpha=0.8),
                            ha='center', va='center', zorder=7)
        
        # Plot reaction forces
        react_idx = 0
        scale = self.get_force_scale()
        for joint, sup_type in self.truss.supports.items():
            x, y = self.truss.joints[joint]
            if sup_type == "pinned":
                rx = reactions[react_idx] if react_idx < len(reactions) else 0
                ry = reactions[react_idx+1] if (react_idx+1) < len(reactions) else 0
                
                if rx != 0:
                    arrow = Arrow(x, y, -rx/abs(rx)*scale, 0, width=scale/2, color='blue', zorder=4)
                    self.ax.add_patch(arrow)
                    self.ax.text(x - rx/abs(rx)*scale*1.2, y, f"Rx={rx:.1f} kN", 
                                bbox=dict(facecolor='white', alpha=0.8), zorder=7)
                
                if ry != 0:
                    arrow = Arrow(x, y, 0, -ry/abs(ry)*scale, width=scale/2, color='blue', zorder=4)
                    self.ax.add_patch(arrow)
                    self.ax.text(x, y - ry/abs(ry)*scale*1.2, f"Ry={ry:.1f} kN", 
                                bbox=dict(facecolor='white', alpha=0.8), zorder=7)
                
                react_idx += 2
            elif sup_type == "roller":
                ry = reactions[react_idx] if react_idx < len(reactions) else 0
                if ry != 0:
                    arrow = Arrow(x, y, 0, -ry/abs(ry)*scale, width=scale/2, color='blue', zorder=4)
                    self.ax.add_patch(arrow)
                    self.ax.text(x, y - ry/abs(ry)*scale*1.2, f"Ry={ry:.1f} kN", 
                                bbox=dict(facecolor='white', alpha=0.8), zorder=7)
                react_idx += 1
        
        self.canvas.draw()
    
    def clear(self):
        self.truss.reset()
        self.ax.clear()
        self.plot_structure()
        self.result_text.delete(1.0, tk.END)
        self.update_button_states()

if __name__ == "__main__":
    root = tk.Tk()
    app = TrussApp(root)
    root.mainloop()