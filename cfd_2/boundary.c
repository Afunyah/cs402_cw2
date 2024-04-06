#include <stdio.h>
#include <string.h>
#include "datadef.h"

#include <mpi.h>
#include <omp.h>

/* Given the boundary conditions defined by the flag matrix, update
 * the u and v velocities. Also enforce the boundary conditions at the
 * edges of the matrix.
 */
void apply_boundary_conditions(float **u, float **v, char **flag,
                               int imax, int jmax, float ui, float vi, int rank, int n_nodes, int initial)
{
    int i, j;

// #pragma omp parallel for schedule(static) private(j)
    for (j = 0; j <= jmax + 1; j++)
    {
        /* Fluid freely flows in from the west */
        if (rank == 0)
        {
            u[0][j] = u[1][j];
            v[0][j] = v[1][j];
        }
        /* Fluid freely flows out to the east */
        if (rank == n_nodes - 1)
        {
            u[imax][j] = u[imax - 1][j];
            v[imax + 1][j] = v[imax][j];
        }
    }

// #pragma omp parallel for schedule(static) private(i)
    for (i = 0; i <= imax + 1; i++)
    {
        /* The vertical velocity approaches 0 at the north and south
         * boundaries, but fluid flows freely in the horizontal direction */
        v[i][jmax] = 0.0;
        u[i][jmax + 1] = u[i][jmax];

        v[i][0] = 0.0;
        u[i][0] = u[i][1];
    }

    MPI_Status status; 
    int node_release_tag = 0; // For ensuring flow from left to right
    int node_update_tag = 1; // For updating boundaries from adjecent ranks

    // Wait for left node to update velocities (flow from left to right)
    if (rank != 0)
    {
        MPI_Recv(u[0], jmax + 2, MPI_FLOAT, rank - 1, node_release_tag, MPI_COMM_WORLD, &status);
        MPI_Recv(v[0], jmax + 2, MPI_FLOAT, rank - 1, node_release_tag, MPI_COMM_WORLD, &status);
    }

    /* Apply no-slip boundary conditions to cells that are adjacent to
     * internal obstacle cells. This forces the u and v velocity to
     * tend towards zero in these cells.
     */
    for (i = 1; i <= imax; i++)
    {
        for (j = 1; j <= jmax; j++)
        {
            if (flag[i][j] & B_NSEW)
            {
                switch (flag[i][j])
                {
                case B_N:
                    v[i][j] = 0.0;
                    u[i][j] = -u[i][j + 1];
                    u[i - 1][j] = -u[i - 1][j + 1];
                    break;
                case B_E:
                    u[i][j] = 0.0;
                    v[i][j] = -v[i + 1][j];
                    v[i][j - 1] = -v[i + 1][j - 1];
                    break;
                case B_S:
                    v[i][j - 1] = 0.0;
                    u[i][j] = -u[i][j - 1];
                    u[i - 1][j] = -u[i - 1][j - 1];
                    break;
                case B_W:
                    u[i - 1][j] = 0.0;
                    v[i][j] = -v[i - 1][j];
                    v[i][j - 1] = -v[i - 1][j - 1];
                    break;
                case B_NE:
                    v[i][j] = 0.0;
                    u[i][j] = 0.0;
                    v[i][j - 1] = -v[i + 1][j - 1];
                    u[i - 1][j] = -u[i - 1][j + 1];
                    break;
                case B_SE:
                    v[i][j - 1] = 0.0;
                    u[i][j] = 0.0;
                    v[i][j] = -v[i + 1][j];
                    u[i - 1][j] = -u[i - 1][j - 1];
                    break;
                case B_SW:
                    v[i][j - 1] = 0.0;
                    u[i - 1][j] = 0.0;
                    v[i][j] = -v[i - 1][j];
                    u[i][j] = -u[i][j - 1];
                    break;
                case B_NW:
                    v[i][j] = 0.0;
                    u[i - 1][j] = 0.0;
                    v[i][j - 1] = -v[i - 1][j - 1];
                    u[i][j] = -u[i][j + 1];
                    break;
                }
            }
        }
    }

    // Send right edge to next node. Releases the node
    if ((rank != n_nodes - 1) && (!initial))
    {
        MPI_Send(u[imax], jmax + 2, MPI_FLOAT, rank + 1, node_release_tag, MPI_COMM_WORLD);
        MPI_Send(v[imax], jmax + 2, MPI_FLOAT, rank + 1, node_release_tag, MPI_COMM_WORLD);
    }

    // Update right edge of previous node.
    // Send to left node
    if (rank != 0)
    {
        MPI_Send(u[1], jmax + 2, MPI_FLOAT, rank - 1, node_update_tag, MPI_COMM_WORLD);
        MPI_Send(v[1], jmax + 2, MPI_FLOAT, rank - 1, node_update_tag, MPI_COMM_WORLD);
    }

    // Receive from right node
    if ((rank != n_nodes - 1) && (!initial))
    {
        MPI_Recv(u[imax + 1], jmax + 2, MPI_FLOAT, rank + 1, node_update_tag, MPI_COMM_WORLD, &status);
        MPI_Recv(v[imax + 1], jmax + 2, MPI_FLOAT, rank + 1, node_update_tag, MPI_COMM_WORLD, &status);
    }

    /* Finally, fix the horizontal velocity at the  western edge to have
     * a continual flow of fluid into the simulation.
     */
    if (rank == 0)
    {
        v[0][0] = 2 * vi - v[1][0];
// #pragma omp parallel for schedule(static) private(j)
        for (j = 1; j <= jmax; j++)
        {
            u[0][j] = ui;
            v[0][j] = 2 * vi - v[1][j];
        }
    }
}
