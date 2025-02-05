void compute_tentative_velocity(float **u, float **v, float **f, float **g,
    char **flag, int imax, int jmax, float del_t, float delx, float dely,
    float gamma, float Re, int rank, int n_nodes, int *disp, int *count_send1, int *count_recv1, int *count_send2, int *count_recv2);

void compute_rhs(float **f, float **g, float **rhs, char **flag, int imax,
    int jmax, float del_t, float delx, float dely);

int poisson(float **p, float **rhs, char **flag, int imax, int jmax,
    float delx, float dely, float eps, int itermax, float omega,
    float *res, int ifull, int rank, int n_nodes, int *sv_disp, int *disp, int *count_send1, int *count_recv1, int *count_send2, int *count_recv2);

void update_velocity(float **u, float **v, float **f, float **g, float **p,
    char **flag, int imax, int jmax, float del_t, float delx, float dely, int rank, int n_nodes);

void set_timestep_interval(float *del_t, int imax, int jmax, float delx,
    float dely, float **u, float **v, float Re, float tau, int rank, int n_nodes);
