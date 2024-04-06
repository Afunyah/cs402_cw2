#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "datadef.h"
#include "init.h"

#include <mpi.h>

#define max(x, y) ((x) > (y) ? (x) : (y))
#define min(x, y) ((x) < (y) ? (x) : (y))

extern int *ileft, *iright;
extern int nprocs, proc;

/* Computation of tentative velocity field (f, g) */
void compute_tentative_velocity(float **u, float **v, float **f, float **g,
                                char **flag, int imax, int jmax, float del_t, float delx, float dely,
                                float gamma, float Re, int rank, int n_nodes, int *disp, int *count_send, int *count_recv)
{
    int i, j;
    float du2dx, duvdy, duvdx, dv2dy, laplu, laplv;

    for (i = 1; i <= imax - 1; i++)
    {
        if (rank == n_nodes - 1 && i == imax)
        {
            continue;
        }
        for (j = 1; j <= jmax; j++)
        {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i + 1][j] & C_F))
            {
                du2dx = ((u[i][j] + u[i + 1][j]) * (u[i][j] + u[i + 1][j]) +
                         gamma * fabs(u[i][j] + u[i + 1][j]) * (u[i][j] - u[i + 1][j]) -
                         (u[i - 1][j] + u[i][j]) * (u[i - 1][j] + u[i][j]) -
                         gamma * fabs(u[i - 1][j] + u[i][j]) * (u[i - 1][j] - u[i][j])) /
                        (4.0 * delx);
                duvdy = ((v[i][j] + v[i + 1][j]) * (u[i][j] + u[i][j + 1]) +
                         gamma * fabs(v[i][j] + v[i + 1][j]) * (u[i][j] - u[i][j + 1]) -
                         (v[i][j - 1] + v[i + 1][j - 1]) * (u[i][j - 1] + u[i][j]) -
                         gamma * fabs(v[i][j - 1] + v[i + 1][j - 1]) * (u[i][j - 1] - u[i][j])) /
                        (4.0 * dely);
                laplu = (u[i + 1][j] - 2.0 * u[i][j] + u[i - 1][j]) / delx / delx +
                        (u[i][j + 1] - 2.0 * u[i][j] + u[i][j - 1]) / dely / dely;

                f[i][j] = u[i][j] + del_t * (laplu / Re - du2dx - duvdy);
            }
            else
            {
                f[i][j] = u[i][j];
            }
        }
    }

    for (i = 1; i <= imax; i++)
    {
        for (j = 1; j <= jmax - 1; j++)
        {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i][j + 1] & C_F))
            {
                duvdx = ((u[i][j] + u[i][j + 1]) * (v[i][j] + v[i + 1][j]) +
                         gamma * fabs(u[i][j] + u[i][j + 1]) * (v[i][j] - v[i + 1][j]) -
                         (u[i - 1][j] + u[i - 1][j + 1]) * (v[i - 1][j] + v[i][j]) -
                         gamma * fabs(u[i - 1][j] + u[i - 1][j + 1]) * (v[i - 1][j] - v[i][j])) /
                        (4.0 * delx);
                dv2dy = ((v[i][j] + v[i][j + 1]) * (v[i][j] + v[i][j + 1]) +
                         gamma * fabs(v[i][j] + v[i][j + 1]) * (v[i][j] - v[i][j + 1]) -
                         (v[i][j - 1] + v[i][j]) * (v[i][j - 1] + v[i][j]) -
                         gamma * fabs(v[i][j - 1] + v[i][j]) * (v[i][j - 1] - v[i][j])) /
                        (4.0 * dely);

                laplv = (v[i + 1][j] - 2.0 * v[i][j] + v[i - 1][j]) / delx / delx +
                        (v[i][j + 1] - 2.0 * v[i][j] + v[i][j - 1]) / dely / dely;

                g[i][j] = v[i][j] + del_t * (laplv / Re - duvdx - dv2dy);
            }
            else
            {
                g[i][j] = v[i][j];
            }
        }
    }



    MPI_Alltoallv(f[1], count_send, disp, MPI_FLOAT, f[imax + 1], count_recv, disp, MPI_FLOAT, MPI_COMM_WORLD);

    MPI_Alltoallv(f[imax], count_send, disp, MPI_FLOAT, f[0], count_recv, disp, MPI_FLOAT, MPI_COMM_WORLD);

    MPI_Alltoallv(g[1], count_send, disp, MPI_FLOAT, g[imax + 1], count_recv, disp, MPI_FLOAT, MPI_COMM_WORLD);

    MPI_Alltoallv(g[imax], count_send, disp, MPI_FLOAT, g[0], count_recv, disp, MPI_FLOAT, MPI_COMM_WORLD);

    /* f & g at external boundaries */

    for (j = 1; j <= jmax; j++)
    {
        if (rank == 0)
        {
            f[imax][j] = u[imax][j];
        }
        if (rank == n_nodes - 1)
        {
            f[imax][j] = u[imax][j];
        }
    }

    for (i = 1; i <= imax; i++)
    {
        g[i][0] = v[i][0];
        g[i][jmax] = v[i][jmax];
    }
}

/* Calculate the right hand side of the pressure equation */
void compute_rhs(float **f, float **g, float **rhs, char **flag, int imax,
                 int jmax, float del_t, float delx, float dely)
{
    int i, j;

    for (i = 1; i <= imax; i++)
    {
        for (j = 1; j <= jmax; j++)
        {
            if (flag[i][j] & C_F)
            {
                /* only for fluid and non-surface cells */
                rhs[i][j] = ((f[i][j] - f[i - 1][j]) / delx +
                             (g[i][j] - g[i][j - 1]) / dely) /
                            del_t;
            }
        }
    }
}

/* Red/Black SOR to solve the poisson equation */
int poisson(float **p, float **rhs, char **flag, int imax, int jmax,
            float delx, float dely, float eps, int itermax, float omega,
            float *res, int ifull, int rank, int n_nodes, int *sv_disp, int *disp, int *count_send, int *count_recv)
{
    int i, j, iter;
    float add, beta_2, beta_mod;
    float p0 = 0.0;

    int rb; /* Red-black value. */

    float rdx2 = 1.0 / (delx * delx);
    float rdy2 = 1.0 / (dely * dely);
    beta_2 = -omega / (2.0 * (rdx2 + rdy2));

    /* Calculate sum of squares */
    for (i = 1; i <= imax; i++)
    {
        for (j = 1; j <= jmax; j++)
        {
            if (flag[i][j] & C_F)
            {
                p0 += p[i][j] * p[i][j];
            }
        }
    }
    float mpi_p0;

    MPI_Reduce(&p0, &mpi_p0, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {

        p0 = sqrt(mpi_p0 / ifull);
        if (p0 < 0.0001)
        {
            p0 = 1.0;
        }
    }

    MPI_Bcast(&p0, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* Red/Black SOR-iteration */
    for (iter = 0; iter < itermax; iter++)
    {
        for (rb = 0; rb <= 1; rb++)
        {
            for (i = 1; i <= imax; i++)
            {
                for (j = 1; j <= jmax; j++)
                {
                    if ((i + j + sv_disp[rank]) % 2 != rb)
                    {
                        continue;
                    }
                    if (flag[i][j] == (C_F | B_NSEW))
                    {
                        /* five point star for interior fluid cells */
                        p[i][j] = (1. - omega) * p[i][j] -
                                  beta_2 * ((p[i + 1][j] + p[i - 1][j]) * rdx2 + (p[i][j + 1] + p[i][j - 1]) * rdy2 - rhs[i][j]);
                    }
                    else if (flag[i][j] & C_F)
                    {
                        /* modified star near boundary */
                        beta_mod = -omega / ((eps_E + eps_W) * rdx2 + (eps_N + eps_S) * rdy2);
                        p[i][j] = (1. - omega) * p[i][j] -
                                  beta_mod * ((eps_E * p[i + 1][j] + eps_W * p[i - 1][j]) * rdx2 + (eps_N * p[i][j + 1] + eps_S * p[i][j - 1]) * rdy2 - rhs[i][j]);
                    }
                } /* end of j */
            }     /* end of i */

            // float *left_col;
            // float *right_col;
            // left_col = (float *)malloc(sizeof(float) * jmax + 2);
            // right_col = (float *)malloc(sizeof(float) * jmax + 2);

            MPI_Alltoallv(p[1], count_send, disp, MPI_FLOAT, p[imax + 1], count_recv, disp, MPI_FLOAT, MPI_COMM_WORLD);

            MPI_Alltoallv(p[imax], count_send, disp, MPI_FLOAT, p[0], count_recv, disp, MPI_FLOAT, MPI_COMM_WORLD);

        } /* end of rb */

        /* Partial computation of residual */
        *res = 0.0;
        for (i = 1; i <= imax; i++)
        {
            for (j = 1; j <= jmax; j++)
            {
                if (flag[i][j] & C_F)
                {
                    /* only fluid cells */
                    add = (eps_E * (p[i + 1][j] - p[i][j]) -
                           eps_W * (p[i][j] - p[i - 1][j])) *
                              rdx2 +
                          (eps_N * (p[i][j + 1] - p[i][j]) -
                           eps_S * (p[i][j] - p[i][j - 1])) *
                              rdy2 -
                          rhs[i][j];
                    *res += add * add;
                }
            }
        }

        float mpi_res;

        MPI_Reduce(res, &mpi_res, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            *res = sqrt((mpi_res) / ifull) / p0;
        }

        MPI_Bcast(res, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

        /* convergence? */
        if (*res < eps)
            break;
    } /* end of iter */

    return iter;
}

/* Update the velocity values based on the tentative
 * velocity values and the new pressure matrix
 */
void update_velocity(float **u, float **v, float **f, float **g, float **p,
                     char **flag, int imax, int jmax, float del_t, float delx, float dely, int rank, int n_nodes)
{
    int i, j;

    for (i = 1; i <= imax - 1; i++)
    {
        if (rank == n_nodes - 1 && i == imax)
        {
            continue;
        }
        for (j = 1; j <= jmax; j++)
        {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i + 1][j] & C_F))
            {
                u[i][j] = f[i][j] - (p[i + 1][j] - p[i][j]) * del_t / delx;
            }
        }
    }
    for (i = 1; i <= imax; i++)
    {
        for (j = 1; j <= jmax - 1; j++)
        {
            /* only if both adjacent cells are fluid cells */
            if ((flag[i][j] & C_F) && (flag[i][j + 1] & C_F))
            {
                v[i][j] = g[i][j] - (p[i][j + 1] - p[i][j]) * del_t / dely;
            }
        }
    }
}

/* Set the timestep size so that we satisfy the Courant-Friedrichs-Lewy
 * conditions (ie no particle moves more than one cell width in one
 * timestep). Otherwise the simulation becomes unstable.
 */
void set_timestep_interval(float *del_t, int imax, int jmax, float delx,
                           float dely, float **u, float **v, float Re, float tau, int rank, int n_nodes)
{
    int i, j;
    float umax, vmax, deltu, deltv, deltRe;

    /* del_t satisfying CFL conditions */
    if (tau >= 1.0e-10)
    { /* else no time stepsize control */
        umax = 1.0e-10;
        vmax = 1.0e-10;
        for (i = 0; i <= imax + 1; i++)
        {
            for (j = 1; j <= jmax + 1; j++)
            {
                umax = max(fabs(u[i][j]), umax);
            }
        }
        for (i = 1; i <= imax + 1; i++)
        {
            for (j = 0; j <= jmax + 1; j++)
            {
                vmax = max(fabs(v[i][j]), vmax);
            }
        }

        float mpi_umax;
        float mpi_vmax;

        MPI_Reduce(&umax, &mpi_umax, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(&vmax, &mpi_vmax, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {
            deltu = delx / mpi_umax;
            deltv = dely / mpi_vmax;
            deltRe = 1 / (1 / (delx * delx) + 1 / (dely * dely)) * Re / 2.0;

            if (deltu < deltv)
            {
                *del_t = min(deltu, deltRe);
            }
            else
            {
                *del_t = min(deltv, deltRe);
            }
            *del_t = tau * (*del_t); /* multiply by safety factor */
        }
        MPI_Bcast(del_t, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
}
