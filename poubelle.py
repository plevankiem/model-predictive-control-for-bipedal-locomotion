
class WalkingFSM:
    def _init_(self, ssp_duration, dsp_duration):
        self.ssp_duration = ssp_duration
        self.dsp_duration = dsp_duration
        self.state = 'Standing'
        self.next_footstep = 2
        self.mcp_timestep = 0.01

        self.start_standing()

    def tick(self):
        if self.state == 'Standing':
            self.run_standing()
        elif self.state == 'DoubleSupport':
            self.run_double_support()
        elif self.state == 'SingleSupport':
            self.run_single_support()
            
    def start_standing(self):
        self.start_walking = False
        self.state = "Standing"
        return self.run_standing()

    def run_standing(self):
        if self.start_walking:
            self.start_walking = False
            if self.next_footstep < len(footsteps):
                return self.start_double_support()
            
    def start_double_support(self):
        if self.next_footstep % 2 == 1:  # left foot swings
            self.stance_foot = "right_foot" #TODO check if stance.right_foot is defined and how to replace it for pinnochio # Stance is the inverse kinematics !!
        else:  # right foot swings
            self.stance_foot = "left_foot"
        dsp_duration = self.dsp_duration
        if self.next_footstep == 2 or self.next_footstep == len(footsteps) - 1:
            # double support is a bit longer for the first and last steps
            dsp_duration = 4 * self.dsp_duration
        self.swing_target = footsteps[self.next_footstep]
        self.rem_time = dsp_duration
        self.state = "DoubleSupport"
        self.start_com_mpc_dsp()
        return self.run_double_support()

    def run_double_support(self):
        if self.rem_time <= 0.:
            return self.start_single_support()
        self.run_com_mpc()
        self.rem_time -= dt

    def start_single_support(self):
        if self.next_footstep % 2 == 1:  # left foot swings
            self.swing_foot = 'left_foot'
        else:  # right foot swings
            self.swing_foot = 'right_foot'
        self.next_footstep += 1
        self.rem_time = self.ssp_duration
        self.state = "SingleSupport"
        self.start_swing_foot()
        self.start_com_mpc_ssp()
        self.run_single_support()

    def run_single_support(self):
        if self.rem_time <= 0.:
            if self.next_footstep < len(footsteps):
                return self.start_double_support()
            else:  # footstep sequence is over
                return self.start_standing()
        self.run_swing_foot()
        self.run_com_mpc()
        self.rem_time -= dt

    def start_swing_foot(self):
        self.swing_start = self.swing_foot.pose

    def start_com_mpc_dsp(self):
        self.update_mpc(self.rem_time, self.ssp_duration)

    def start_com_mpc_ssp(self):
        self.update_mpc(0., self.rem_time)

    def run_com_mpc(self):
            if self.preview_time >= self.mpc_timestep:
                if self.state == "DoubleSupport":
                    self.update_mpc(self.rem_time, self.ssp_duration)
                else:  # self.state == "SingleSupport":
                    self.update_mpc(0., self.rem_time)
            com_jerk = np.array([self.x_mpc.U[0][0], self.y_mpc.U[0][0], 0.]) #TODO update here 
            #stance.com.integrate_constant_jerk(com_jerk, dt) # Integrate CoM with the computed jerk
            self.preview_time += dt

    def run_swing_foot(self):
        progress = min(1., max(0., 1. - self.rem_time / self.ssp_duration))
        new_pose = pin.interpolate(model, self.swing_start, self.swing_target.pose, progress)
        self.swing_foot.set_pose(new_pose)

    #TODO change the implementation of generate_footsteps to use the paper !
    def update_mpc(self):
        nb_preview_steps = 16
        T = self.mpc_timestep
        nb_init_dsp_steps = int(round(self.dsp_duration / T))
        nb_init_ssp_steps = int(round(self.ssp_duration / T))
        nb_dsp_steps = int(round(self.dsp_duration / T))
        A = np.array([[1., T, T ** 2 / 2.], [0., 1., T], [0., 0., 1.]])
        B = np.array([T * 3 / 6., T * 2 / 2., T]).reshape((3, 1))
        h = 0.75
        g = 9.81
        zmp_from_state = np.array([1., 0., -h / g])
        C = np.array([+zmp_from_state, -zmp_from_state])
        e = [[], []]
        cur_vertices = self.stance_foot.get_scaled_contact_area(0.8)
        next_vertices = self.swing_target.get_scaled_contact_area(0.8)
        for coord in [0, 1]:
            cur_max = max(v[coord] for v in cur_vertices)
            cur_min = min(v[coord] for v in cur_vertices)
            next_max = max(v[coord] for v in next_vertices)
            next_min = min(v[coord] for v in next_vertices)
            e[coord] = [
                np.array([+1000., +1000.]) if i < nb_init_dsp_steps else
                np.array([+cur_max, -cur_min]) if i - nb_init_dsp_steps <= nb_init_ssp_steps else
                np.array([+1000., +1000.]) if i - nb_init_dsp_steps - nb_init_ssp_steps < nb_dsp_steps else
                np.array([+next_max, -next_min])
                for i in range(nb_preview_steps)]
        x_mpc = LinearPredictiveControl(
            A, B, C, e[0],
            x_init=np.array([self.x_hist[-1]]),
            nb_steps=nb_preview_steps,
            wxt=1., wu=0.01)
        y_mpc = LinearPredictiveControl(
            A, B, C, e[1],
            x_init=np.array([self.y_hist[-1]]),
            nb_steps=nb_preview_steps,
            wxt=1., wu=0.01)
        self.x_hist.append(x_mpc)
        self.y_hist.append(y_mpc)
        self.preview_time = 0.