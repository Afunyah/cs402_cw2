#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <errno.h>
#include <stddef.h>
#include "alloc.h"
#include "boundary.h"
#include "datadef.h"
#include "init.h"
#include "simulation.h"

#include <mpi.h>

void write_bin(float **u, float **v, float **p, char **flag,
               int imax, int jmax, float xlength, float ylength, char *file);

int read_bin(float **u, float **v, float **p, char **flag,
             int imax, int jmax, float xlength, float ylength, char *file);

static void print_usage(void);
static void print_version(void);
static void print_help(void);

static char *progname;

int proc = 0;   /* Rank of the current process */
int nprocs = 0; /* Number of processes in communicator */

int *ileft, *iright; /* Array bounds for each processor */

#define PACKAGE "karman"
#define VERSION "1.0"

/* Command line options */
static struct option long_opts[] = {
    {"del-t", 1, NULL, 'd'},
    {"help", 0, NULL, 'h'},
    {"imax", 1, NULL, 'x'},
    {"infile", 1, NULL, 'i'},
    {"jmax", 1, NULL, 'y'},
    {"outfile", 1, NULL, 'o'},
    {"t-end", 1, NULL, 't'},
    {"verbose", 1, NULL, 'v'},
    {"version", 1, NULL, 'V'},
    {0, 0, 0, 0}};
#define GETOPTS "d:hi:o:t:v:Vx:y:"

int main(int argc, char *argv[])
{
    int verbose = 2;      /* Verbosity level */
    float xlength = 22.0; /* Width of simulated domain */
    float ylength = 4.1;  /* Height of simulated domain */
    int imax = 660;       /* Number of cells horizontally */
    int jmax = 120;       /* Number of cells vertically */

    char *infile;  /* Input raw initial conditions */
    char *outfile; /* Output raw simulation results */

    float t_end = 2.1;   /* Simulation runtime */
    float del_t = 0.003; /* Duration of each timestep */
    float tau = 0.5;     /* Safety factor for timestep control */

    int itermax = 100; /* Maximum number of iterations in SOR */
    float eps = 0.001; /* Stopping error threshold for SOR */
    float omega = 1.7; /* Relaxation parameter for SOR */
    float gamma = 0.9; /* Upwind differencing factor in PDE
                          discretisation */

    float Re = 150.0; /* Reynolds number */
    float ui = 1.0;   /* Initial X velocity */
    float vi = 0.0;   /* Initial Y velocity */

    float t, delx, dely;
    int i, j, itersor = 0, ifluid = 0, ibound = 0;
    float res;
    float **u, **v, **p, **rhs, **f, **g;
    char **flag;
    int init_case, iters = 0;
    int show_help = 0, show_usage = 0, show_version = 0;

    progname = argv[0];
    infile = strdup("karman.bin");
    outfile = strdup("karman.bin");

    int optc;
    while ((optc = getopt_long(argc, argv, GETOPTS, long_opts, NULL)) != -1)
    {
        switch (optc)
        {
        case 'h':
            show_help = 1;
            break;
        case 'V':
            show_version = 1;
            break;
        case 'v':
            verbose = atoi(optarg);
            break;
        case 'x':
            imax = atoi(optarg);
            break;
        case 'y':
            jmax = atoi(optarg);
            break;
        case 'i':
            free(infile);
            infile = strdup(optarg);
            break;
        case 'o':
            free(outfile);
            outfile = strdup(optarg);
            break;
        case 'd':
            del_t = atof(optarg);
            break;
        case 't':
            t_end = atof(optarg);
            break;
        default:
            show_usage = 1;
        }
    }
    if (show_usage || optind < argc)
    {
        print_usage();
        return 1;
    }

    if (show_version)
    {
        print_version();
        if (!show_help)
        {
            return 0;
        }
    }

    if (show_help)
    {
        print_help();
        return 0;
    }

    int rank;    // My rank
    int n_nodes; // Total number of processes
    MPI_Status status;

    int *sv_disp;
    int *sv_disp2;
    int *i_width_arr;
    int *i_width_arr_exp;

    float **u_final, **v_final, **p_final;
    char **flag_final;

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_nodes);

    printf("Hello world, from process %d of %d\n", rank, n_nodes);

    delx = xlength / imax;
    dely = ylength / jmax;
    int imax_node = 0;

    // struct read_dat
    // {
    //     float u_read[jmax + 2];
    //     float v_read[jmax + 2];
    //     float p_read[jmax + 2];
    //     char flag_read[jmax + 2];
    // };

    if (rank == 0)
    {
        i_width_arr = (int *)calloc(n_nodes, sizeof(int));
        i_width_arr_exp = (int *)calloc(n_nodes, sizeof(int));
        // Perform this calculation once in root node
        int m = 0;
        // Tally to account for uneven number of nodes, where the problem size is not equally divided
        for (int k = 1; k <= imax; k++)
        {
            if (m == n_nodes)
            {
                m = 0;
            }
            i_width_arr[m] += 1;
            m += 1;
        }

        // Additional columns for node boundaries, stored in root rank
        for (int k = 0; k < n_nodes; k++)
        {
            i_width_arr_exp[k] = (i_width_arr[k]+2)*(jmax+2);
            // i_width_arr_exp[k] = (i_width_arr[k])*(jmax);
            printf("i_width_arr %d = %d\n", k, i_width_arr[k]);
        }

        MPI_Scatter(i_width_arr, 1, MPI_INT, &imax_node, 1, MPI_INT, 0, MPI_COMM_WORLD); // Send imax of each node
    }
    else
    {
        MPI_Scatter(NULL, 1, MPI_INT, &imax_node, 1, MPI_INT, 0, MPI_COMM_WORLD); // Send imax of each node
    }

    printf("imax node =  %d\n", imax_node);

    // /* Allocate arrays */
    // // The size of the i array will include 2 buffers
    // // This will be used by each node to store data from other nodes and boundaries
    u = alloc_floatmatrix(imax_node + 2, jmax + 2);
    v = alloc_floatmatrix(imax_node + 2, jmax + 2);
    f = alloc_floatmatrix(imax_node + 2, jmax + 2);
    g = alloc_floatmatrix(imax_node + 2, jmax + 2);
    p = alloc_floatmatrix(imax_node + 2, jmax + 2);
    rhs = alloc_floatmatrix(imax_node + 2, jmax + 2);
    flag = alloc_charmatrix(imax_node + 2, jmax + 2);

    // if (!u || !v || !f || !g || !p || !rhs || !flag)
    // {
    //     fprintf(stderr, "Rank %d\n", rank);
    //     fprintf(stderr, "Couldn't allocate memory for matrices h.\n");

    //     return 1;
    // }

    // MPI_Datatype MPI_FLOATARRAY;
    // MPI_Type_contiguous(jmax + 2, MPI_FLOAT, &MPI_FLOATARRAY);
    // MPI_Type_commit(&MPI_FLOATARRAY);

    // MPI_Datatype MPI_CHARARRAY;
    // MPI_Type_contiguous(jmax + 2, MPI_CHAR, &MPI_CHARARRAY);
    // MPI_Type_commit(&MPI_CHARARRAY);

    // // /* https://rookiehpc.org/mpi/docs/mpi_type_create_struct/index.html */
    // MPI_Datatype read_bin_type;

    // int lengths[4] = {1, 1, 1, 1};
    // // int lengths[4] = {jmax+2, jmax+2, jmax+2, jmax+2};

    // MPI_Aint displacements[] = {offsetof(struct read_dat, u_read), offsetof(struct read_dat, v_read), offsetof(struct read_dat, p_read), offsetof(struct read_dat, flag_read)};

    // MPI_Datatype types[4] = {MPI_FLOATARRAY, MPI_FLOATARRAY, MPI_FLOATARRAY, MPI_CHARARRAY};
    // // MPI_Datatype types[4] = {MPI_FLOAT, MPI_FLOAT, MPI_FLOAT, MPI_CHAR};
    // MPI_Type_create_struct(4, lengths, displacements, types, &read_bin_type);

    // MPI_Type_commit(&read_bin_type);

    init_case = 0;
    sv_disp = (int *)calloc(n_nodes, sizeof(int));
    if (rank == 0)
    {
        int sum = 0;
        for (int i = 1; i < n_nodes; i++)
        {
            sum = sum + i_width_arr[i - 1];
            sv_disp[i] = sum;
            // printf("sv_disp %d %d\n", i, sv_disp[i]);
        }
    }
    MPI_Bcast(sv_disp, n_nodes, MPI_INT, 0, MPI_COMM_WORLD);
    printf("sv_disp %d %d\n", rank, sv_disp[rank]);

    sv_disp2 = (int *)calloc(n_nodes, sizeof(int));
    if (rank == 0)
    {
        int sum = 0;
        for (int i = 1; i < n_nodes; i++)
        {
            sum = sum + i_width_arr[i - 1] * (jmax + 2);
            sv_disp2[i] = sum;
            // printf("sv_disp %d %d\n", i, sv_disp[i]);
        }
    }
    MPI_Bcast(sv_disp2, n_nodes, MPI_INT, 0, MPI_COMM_WORLD);

    // MPI_File fh;
    // MPI_Offset offset;
    // // MPI_Status status;

    // MPI_Barrier(MPI_COMM_WORLD);
    // int rbtsz;
    // MPI_Type_size(read_bin_type, &rbtsz);
    // // offset = (MPI_Offset)(sizeof(int) * 4 + sv_disp[rank] * rbtsz);
    // offset = (MPI_Offset)(sizeof(int) * 2 + sizeof(float) * 2);

    // printf("rbtsz %d\n", rbtsz);

    // MPI_File_open(MPI_COMM_WORLD, infile, MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
    // MPI_File_set_view(fh, offset, read_bin_type, read_bin_type, "native", MPI_INFO_NULL);
    // // struct read_dat *uvpflag = (struct read_dat *) malloc ((imax_node+2)*sizeof(struct read_dat));
    // struct read_dat uvpflag;
    // // uvpflag = (struct read_dat *)calloc(1, sizeof(struct read_dat));

    // for (int i = 0; i < imax_node + 2; i++)
    // {
    //     // MPI_Barrier(MPI_COMM_WORLD);

    //     MPI_File_read(fh, &uvpflag, 1, read_bin_type, &status);
    //     // MPI_File_read_all(fh, u[i], jmax + 2, MPI_FLOAT, &status);
    //     // MPI_File_read_all(fh, v[i], jmax + 2, MPI_FLOAT, &status);
    //     // MPI_File_read_all(fh, p[i], jmax + 2, MPI_FLOAT, &status);
    //     // MPI_File_read_all(fh, flag[i], jmax + 2, MPI_CHAR, &status);
    //     if (rank == 0)
    //     {
    //         printf("HEREE %f\n", uvpflag.u_read[0]);
    //         //  int count = 0;
    //         // MPI_Get_count(&status, read_bin_type, &count);
    //         // printf("(%d) on read\n", count);
    //     }
    //     u[i] = uvpflag.u_read;
    //     v[i] = uvpflag.v_read;
    //     p[i] = uvpflag.p_read;
    //     flag[i] = uvpflag.flag_read;
    // }
    // // printf("HEREE %f\n", u[0]);

    // MPI_Barrier(MPI_COMM_WORLD);
    // MPI_File_close(&fh);

    // printf("READ into nodes %d %f\n", rank, u[11][11]);
    float **u_full, **v_full, **p_full;
    char **flag_full;
    if (rank == 0)
    {
        u_full = alloc_floatmatrix(imax + 2, jmax + 2);
        v_full = alloc_floatmatrix(imax + 2, jmax + 2);
        p_full = alloc_floatmatrix(imax + 2, jmax + 2);
        flag_full = alloc_charmatrix(imax + 2, jmax + 2);

        if (!u_full || !v_full || !p_full || !flag_full)
        {
            fprintf(stderr, "Full grid array %d\n", rank);
            fprintf(stderr, "Couldn't allocate memory for matrices.\n");

            return 1;
        }

        /* Read in initial values from a file if it exists */
        init_case = read_bin(u_full, v_full, p_full, flag_full, imax, jmax, xlength, ylength, infile);
        // init_case = -1;

        if (init_case > 0)
        {
            /* Error while reading file */
            return 1;
        }
        printf("HERE -2 %d\n", rank);
        if (init_case < 0)
        {
            printf("HERE -1 %d\n", rank);
            /* Set initial values if file doesn't exist */
            for (i = 0; i < imax + 2; i++)
            {
                for (j = 0; j < jmax + 2; j++)
                {
                    u_full[i][j] = ui;
                    v_full[i][j] = vi;
                    p_full[i][j] = 0.0;
                }
            }
            printf("HERE 10 %d\n", rank);
            init_flag(flag_full, imax, jmax, delx, dely, &ibound, rank, n_nodes);
            printf("HERE 00 %d\n", rank);
            apply_boundary_conditions(u_full, v_full, flag_full, imax, jmax, ui, vi, rank, n_nodes, 1, 0);
            printf("HERE -00 %d\n", rank);
        }
        if (init_case == 0)
        {
        }

        // printf("HERE +1 %d\n", rank);
        // for (int n = 1; n < n_nodes; n++)
        // {
        //     // now need to construct the array specifically for the ith MPI node
        //     for (i = 0; i < imax_node + 2; i++)
        //     {
        //         int offset = sv_disp[n] + i;
        //         for (j = 0; j < jmax + 2; j++)
        //         {
        //             u[i][j] = u_full[offset][j];
        //             v[i][j] = v_full[offset][j];
        //             p[i][j] = p_full[offset][j];
        //             flag[i][j] = flag_full[offset][j];
        //         }
        //     }
        //     // Send the four necessary arrays

        //     for (i = 0; i < imax_node + 2; i++)
        //     {  
        //         MPI_Send(u[i], jmax + 2, MPI_FLOAT, n, 0, MPI_COMM_WORLD);
        //         MPI_Send(v[i], jmax + 2, MPI_FLOAT, n, 0, MPI_COMM_WORLD);
        //         MPI_Send(p[i], jmax + 2, MPI_FLOAT, n, 0, MPI_COMM_WORLD);
        //         MPI_Send(flag[i], jmax + 2, MPI_CHAR, n, 0, MPI_COMM_WORLD);
        //     }
        // }

        // printf("HERE +2 %d\n", rank);
        // // Finally, fill in our own array
        // for (i = 0; i <= imax_node + 1; i++)
        // {
        //     for (j = 0; j <= jmax + 1; j++)
        //     {
        //         u[i][j] = u_full[i][j];
        //         v[i][j] = v_full[i][j];
        //         p[i][j] = p_full[i][j];
        //         flag[i][j] = flag_full[i][j];
        //     }
        // }
        printf("HERE +3 %d\n", rank);
    }
    // else
    // {
    //     printf("HERE +4 %d\n", rank);
    //     // printf("Node %d is still active having finished the handshake\n", rank);

    //     // Reaching this point means the handshake has been completed
    //     // need to loop through and get the columns separately!
    //     for (i = 0; i < imax_node + 2; i++)
    //     {
    //         MPI_Recv(u[i], jmax + 2, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
    //         MPI_Recv(v[i], jmax + 2, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
    //         MPI_Recv(p[i], jmax + 2, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &status);
    //         MPI_Recv(flag[i], jmax + 2, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
    //         // Debug line
    //         // printf("Node %d successfully received round %d of %d array values\n",rank, i+1, imaxNode+2);
    //     }
    // }
    // MPI_Datatype MPI_FLOATARRAY;
    // MPI_Type_contiguous(jmax, MPI_FLOAT, &MPI_FLOATARRAY);
    // MPI_Type_commit(&MPI_FLOATARRAY);
    printf("HERE x+4 %d\n", rank);
    if (rank == 0)
    {
        MPI_Scatterv(&u_full[0][0], i_width_arr_exp, sv_disp2, MPI_FLOAT, &u[0][0], (imax_node+2)*(jmax+2), MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(&v_full[0][0], i_width_arr_exp, sv_disp2, MPI_FLOAT, &v[0][0], (imax_node+2)*(jmax+2), MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(&p_full[0][0], i_width_arr_exp, sv_disp2, MPI_FLOAT, &p[0][0], (imax_node+2)*(jmax+2), MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(&flag_full[0][0], i_width_arr_exp, sv_disp2, MPI_CHAR, &flag[0][0], (imax_node+2)*(jmax+2), MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, &u[0][0], (imax_node+2)*(jmax+2), MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, &v[0][0], (imax_node+2)*(jmax+2), MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, &p[0][0], (imax_node+2)*(jmax+2), MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, &flag[0][0], (imax_node+2)*(jmax+2), MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    printf("HERE +5 %d\n", rank);
    // MPI_Barrier(MPI_COMM_WORLD);
    // int sv_disp[0];

    // float **u_full, **v_full, **p_full;
    // char **flag_full;
    // /* Allocate arrays for full grid */
    // u_full = alloc_floatmatrix(imax + 2, jmax + 2);
    // v_full = alloc_floatmatrix(imax + 2, jmax + 2);
    // p_full = alloc_floatmatrix(imax + 2, jmax + 2);
    // flag_full = alloc_charmatrix(imax + 2, jmax + 2);
    // if (rank == 0)
    // {
    //     // float **u_full, **v_full, **p_full;
    //     // char **flag_full;
    //     // /* Allocate arrays for full grid */
    //     // u_full = alloc_floatmatrix(imax + 2, jmax + 2);
    //     // v_full = alloc_floatmatrix(imax + 2, jmax + 2);
    //     // p_full = alloc_floatmatrix(imax + 2, jmax + 2);
    //     // flag_full = alloc_charmatrix(imax + 2, jmax + 2);

    //     if (!u_full || !v_full || !p_full || !flag_full)
    //     {
    //         fprintf(stderr, "Full grid array %d\n", rank);
    //         fprintf(stderr, "Couldn't allocate memory for matrices.\n");

    //         return 1;
    //     }

    //     /* Read in initial values from a file if it exists */
    //     init_case = read_bin(u_full, v_full, p_full, flag_full, imax, jmax, xlength, ylength, infile);

    //     // Displacement array for mpi_scatterv (and gatherv) stores offsets
    //     // int sv_disp[n_nodes];
    //     sv_disp = malloc(n_nodes * sizeof(int));
    //     sv_disp[0] = 0;

    //     int sum = 0; // Subtract 1 to capture left column from left node
    //     for (int i = 1; i < n_nodes; i++)
    //     {
    //         sum = sum + i_width_arr[i - 1];
    //         sv_disp[i] = sum;
    //     }

    //     // If the file exists, correctly
    //     if (init_case == 0)
    //     {
    //         // Send out the u,v,p and flag arrays to each node.
    //         // The full grid array is split based on the i_widths and displacements from the 0th column
    //         // MPI_Scatterv(u_full, i_width_arr_exp, sv_disp, MPI_FLOAT, u, imax_node + 2, MPI_FLOAT, 0, MPI_COMM_WORLD);
    //         // MPI_Scatterv(v_full, i_width_arr_exp, sv_disp, MPI_FLOAT, v, imax_node + 2, MPI_FLOAT, 0, MPI_COMM_WORLD);
    //         // MPI_Scatterv(p_full, i_width_arr_exp, sv_disp, MPI_FLOAT, p, imax_node + 2, MPI_FLOAT, 0, MPI_COMM_WORLD);
    //         // MPI_Scatterv(flag_full, i_width_arr_exp, sv_disp, MPI_CHAR, flag, imax_node + 2, MPI_CHAR, 0, MPI_COMM_WORLD);
    //     }

    //     if (init_case > 0)
    //     {
    //         /* Error while reading file */
    //         return 1;
    //     }

    //     // for (int i = 0; i < 100; i++)
    //     // {
    //     //     printf("%f\n", u_full[331][i]);
    //     // }

    //     MPI_Bcast(&init_case, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // }
    // else
    // {
    //     MPI_Bcast(&init_case, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //     if (init_case == 0)
    //     {
    //         // MPI_Scatterv(u_full, i_width_arr_exp, sv_disp, MPI_FLOAT, u, imax_node + 2, MPI_FLOAT, 0, MPI_COMM_WORLD);
    //         // MPI_Scatterv(v_full, i_width_arr_exp, sv_disp, MPI_FLOAT, v, imax_node + 2, MPI_FLOAT, 0, MPI_COMM_WORLD);
    //         // MPI_Scatterv(p_full, i_width_arr_exp, sv_disp, MPI_FLOAT, p, imax_node + 2, MPI_FLOAT, 0, MPI_COMM_WORLD);
    //         // MPI_Scatterv(flag_full, i_width_arr_exp, sv_disp, MPI_CHAR, flag, imax_node + 2, MPI_CHAR, 0, MPI_COMM_WORLD);

    //         // for (int i = 0; i < 5; i++)
    //         // {
    //         //     printf("%f\n", u[0][i]);
    //         // }
    //     }

    //     // if (init_case == 0)
    //     // {
    //     //     MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, u, imax_node + 2, MPI_FLOAT, 0, MPI_COMM_WORLD);
    //     //     MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, v, imax_node + 2, MPI_FLOAT, 0, MPI_COMM_WORLD);
    //     //     MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, p, imax_node + 2, MPI_FLOAT, 0, MPI_COMM_WORLD);
    //     //     MPI_Scatterv(NULL, NULL, NULL, MPI_CHAR, flag, imax_node + 2, MPI_CHAR, 0, MPI_COMM_WORLD);
    //     // }
    // }

    // If there is no initial state
    // if (init_case < 0)
    // {
    //     /* Set initial values if file doesn't exist */
    //     for (i = 0; i <= imax_node + 1; i++)
    //     {
    //         for (j = 0; j <= jmax + 1; j++)
    //         {
    //             u[i][j] = ui;
    //             v[i][j] = vi;
    //             p[i][j] = 0.0;
    //         }
    //     }
    //     init_flag(flag, imax_node, jmax, delx, dely, &ibound, rank, n_nodes);
    //     apply_boundary_conditions(u, v, flag, imax_node, jmax, ui, vi, rank, n_nodes);
    // }
    MPI_Barrier(MPI_COMM_WORLD);
    /* Main loop */
    for (t = 0.0; t < t_end; t += del_t, iters++)
    {
        set_timestep_interval(&del_t, imax_node, jmax, delx, dely, u, v, Re, tau, rank, n_nodes);
        // printf("HERE +6 %d\n", rank);
        ifluid = (imax * jmax) - ibound;

        compute_tentative_velocity(u, v, f, g, flag, imax_node, jmax,
                                   del_t, delx, dely, gamma, Re, rank, n_nodes);
        // printf("HERE +7 %d\n", rank);
        compute_rhs(f, g, rhs, flag, imax_node, jmax, del_t, delx, dely);

        if (ifluid > 0)
        {
            itersor = poisson(p, rhs, flag, imax_node, jmax, delx, dely,
                              eps, itermax, omega, &res, ifluid, rank, n_nodes, sv_disp);
        }
        else
        {
            itersor = 0;
        }
        // printf("HERE +8 %d\n", rank);

        // if (rank == 1 && verbose > 1)
        {
            printf("%d t:%g, del_t:%g, SOR iters:%3d, res:%e, bcells:%d\n",
                   iters, t + del_t, del_t, itersor, res, ibound);
        }

        update_velocity(u, v, f, g, p, flag, imax_node, jmax, del_t, delx, dely, rank, n_nodes);
        // printf("HERE +9 %d\n", rank);
        apply_boundary_conditions(u, v, flag, imax_node, jmax, ui, vi, rank, n_nodes, 0, iters);
        // MPI_Barrier(MPI_COMM_WORLD);
        // printf("HERE +10 %d\n", rank);
    } /* End of main loop */

    printf("HERE +++ %d\n", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    // printf("Here 00 %d\n", rank);
    // // Commit a MPI_Datatype for these arrays.
    // MPI_Datatype MPI_FLOATARRAY;
    // MPI_Type_contiguous(imax_node+2, MPI_FLOAT, &MPI_FLOATARRAY);
    // MPI_Type_commit(&MPI_FLOATARRAY);

    // MPI_Datatype MPI_CHARARRAY;
    // MPI_Type_contiguous(imax_node+2, MPI_CHAR, &MPI_CHARARRAY);
    // MPI_Type_commit(&MPI_CHARARRAY);

    // // Displacements for root node only. The root node scatters the data
    // if (rank == 0)
    // {

    //     /* Allocate arrays for full grid */
    //     u_final = alloc_floatmatrix(imax + 2, jmax + 2);
    //     v_final = alloc_floatmatrix(imax + 2, jmax + 2);
    //     p_final = alloc_floatmatrix(imax + 2, jmax + 2);
    //     flag_final = alloc_charmatrix(imax + 2, jmax + 2);

    //     if (!u_final || !v_final || !p_final || !flag_final)
    //     {
    //         fprintf(stderr, "Full grid array %d\n", rank);
    //         fprintf(stderr, "Couldn't allocate memory for matrices.\n");

    //         return 1;
    //     }
    //     // Displacement array for mpi_scatterv stores offsets
    //     // int sv_disp[n_nodes];
    //     // sv_disp[0] = 0;

    //     // int sum = -1; // Subtract 1 to capture left column from left node
    //     // for (int i = 1; i < n_nodes; i++)
    //     // {
    //     //     sum = sum + i_width_arr[i - 1];
    //     //     sv_disp[i] = sum;
    //     // }

    //     // for (int k = 0; k < sizeof(i_width_arr_exp) / sizeof(int); k++)
    //     // {
    //     //     printf("i_width_arr_exp %d = %d\n", k, i_width_arr_exp[k]);
    //     // }
    //     printf("HELLO\n");
    //     // printf("size of sv_disp = %lu\n", sizeof(sv_disp));
    //     for (int k = 0; k < n_nodes; k++)
    //     {
    //         printf("111 sv_disp %d = %d\n", k, sv_disp[k]);
    //     }

    //     // First 3 arguments of scatterv are NULL for receiving nodes!
    //     // Send out the u,v,p and flag arrays to each node.
    //     // The full grid array is split based on the i_widths and displacements from the 0th column
    //     MPI_Gatherv(u[0], imax_node + 2, MPI_FLOATARRAY, u_final[0], i_width_arr_exp, sv_disp, MPI_FLOATARRAY, 0, MPI_COMM_WORLD);
    //     // MPI_Gatherv(v[0], imax_node + 2, MPI_FLOATARRAY, v_final[0], i_width_arr_exp, sv_disp, MPI_FLOATARRAY, 0, MPI_COMM_WORLD);
    //     // MPI_Gatherv(p[0], imax_node + 2, MPI_FLOATARRAY, p_final[0], i_width_arr_exp, sv_disp, MPI_FLOATARRAY, 0, MPI_COMM_WORLD);
    //     // MPI_Gatherv(flag[0], imax_node + 2, MPI_CHARARRAY, flag_final[0], i_width_arr_exp, sv_disp, MPI_CHARARRAY, 0, MPI_COMM_WORLD);
    // }
    // else
    // {
    //     MPI_Gatherv(u[0], imax_node + 2, MPI_FLOATARRAY, NULL, NULL, NULL, MPI_FLOATARRAY, 0, MPI_COMM_WORLD);
    //     // MPI_Gatherv(v[0], imax_node + 2, MPI_FLOATARRAY, NULL, NULL, NULL, MPI_FLOATARRAY, 0, MPI_COMM_WORLD);
    //     // MPI_Gatherv(p[0], imax_node + 2, MPI_FLOATARRAY, NULL, NULL, NULL, MPI_FLOATARRAY, 0, MPI_COMM_WORLD);
    //     // MPI_Gatherv(flag[0], imax_node + 2, MPI_CHARARRAY, NULL, NULL, NULL, MPI_CHARARRAY, 0, MPI_COMM_WORLD);
    // }
    // printf("Here 11 %d\n", rank);
    // MPI_Barrier(MPI_COMM_WORLD);
    // if (rank == 0)
    // {
    //     for (int i = 0; i < 20; i++)
    //     {
    //         printf("%f\n", u_final[299][i]);
    //     }

    //     // if (outfile != NULL && strcmp(outfile, "") != 0 && proc == 0)
    //     // {
    //     //     write_bin(u_final, v_final, p_final, flag_final, imax, jmax, xlength, ylength, outfile);
    //     // }
    // }
    // printf("Here 22 %d\n", rank);
    // MPI_Barrier(MPI_COMM_WORLD);
    // // MPI_File fh1;
    // offset = (MPI_Offset)(sizeof(float) * 1 * rank);
    // // printf("Here 1 %d\n", rank);
    // // MPI_File_set_view(fh, offset, read_bin_type, read_bin_type, "native", MPI_INFO_NULL);
    // MPI_File_open(MPI_COMM_WORLD, outfile, MPI_MODE_RDWR, MPI_INFO_NULL, &fh);
    // // printf("Here 2 %d\n", rank);

    // // printf("Here 3 %d\n", rank);
    // // MPI_File_set_view(fh, offset, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);
    // // printf("Here 4 %d\n", rank);
    // // MPI_File_write_all(fh, u[0], 100, MPI_FLOAT, &status);
    // // MPI_Barrier(MPI_COMM_WORLD);
    // // MPI_File_write_all(fh, v[0], 100, MPI_FLOAT, &status);
    // // MPI_Barrier(MPI_COMM_WORLD);
    // // MPI_File_write_all(fh, p[0], 100, MPI_FLOAT, &status);
    // // MPI_Barrier(MPI_COMM_WORLD);
    // // MPI_File_set_view(fh, offset, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL);
    // // MPI_File_write_all(fh, flag[0], 100, MPI_CHAR, &status);
    // // MPI_File_write_all(fh, v, jmax + 2, MPI_FLOAT, &status);
    // // MPI_File_write_all(fh, p, jmax + 2, MPI_FLOAT, &status);
    // // MPI_File_write_all(fh, flag, jmax + 2, MPI_CHAR, &status);
    // for (int i = 0; i < 10; i++)
    // {
    //     // printf("Here 5 %d", rank);
    //     // MPI_File_write(fh, u[i], 1, MPI_FLOAT, &status);
    //     // MPI_File_write_all(fh, v[i], jmax + 2, MPI_FLOAT, &status);
    //     // MPI_File_write_all(fh, p[i], jmax + 2, MPI_FLOAT, &status);
    //     // MPI_File_write_all(fh, flag[i], jmax + 2, MPI_CHAR, &status);
    //     // printf("Here 6 %d", rank);
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     MPI_File_set_view(fh, offset, MPI_FLOAT, MPI_FLOAT, "native", MPI_INFO_NULL);
    //     MPI_File_write_all(fh, u[i], 100, MPI_FLOAT, &status);
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     MPI_File_write_all(fh, v[i], 100, MPI_FLOAT, &status);
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     MPI_File_write_all(fh, p[i], 100, MPI_FLOAT, &status);
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     MPI_File_set_view(fh, offset, MPI_CHAR, MPI_CHAR, "native", MPI_INFO_NULL);
    //     MPI_File_write_all(fh, flag[i], 100, MPI_CHAR, &status);
    // }
    // printf("Here 7 %d\n", rank);
    // // MPI_Barrier(MPI_COMM_WORLD);
    // MPI_File_close(&fh);

    // free_matrix(u);
    // free_matrix(v);
    // free_matrix(f);
    // free_matrix(g);
    // free_matrix(p);
    // free_matrix(rhs);
    // free_matrix(flag);
    // MPI_Barrier(MPI_COMM_WORLD);
    printf("HERE to write %d\n", rank);
    // Need to collate all of the data
    if (rank == 0)
    {
        /* Allocate arrays for full grid */
        u_final = alloc_floatmatrix(imax + 2, jmax + 2);
        v_final = alloc_floatmatrix(imax + 2, jmax + 2);
        p_final = alloc_floatmatrix(imax + 2, jmax + 2);
        flag_final = alloc_charmatrix(imax + 2, jmax + 2);

        // Make sure to write the root information to the larger temp array as well
        for (i = 0; i < imax_node + 2; i++)
        {
            for (j = 0; j < jmax + 2; j++)
            {
                u_final[i][j] = u[i][j];
                v_final[i][j] = v[i][j];
                p_final[i][j] = p[i][j];
                flag_final[i][j] = flag[i][j];
            }
        }

        printf("Here 5 %d\n", rank);
        for (int n = 1; n < n_nodes; n++)
        {
            for (i = 0; i < imax_node + 2; i++)
            {
                int offset = sv_disp[n] + i;
                MPI_Recv(u_final[offset], jmax + 2, MPI_FLOAT, n, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(v_final[offset], jmax + 2, MPI_FLOAT, n, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(p_final[offset], jmax + 2, MPI_FLOAT, n, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(flag_final[offset], jmax + 2, MPI_CHAR, n, 0, MPI_COMM_WORLD, &status);
            }
            printf("Here 4 %d\n", rank);
        }
    }
    else
    {

        for (i = 0; i < imax_node + 2; i++)
        {
            MPI_Send(u[i], jmax + 2, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(v[i], jmax + 2, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(p[i], jmax + 2, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(flag[i], jmax + 2, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
    {
        if (outfile != NULL && strcmp(outfile, "") != 0 && proc == 0)
        {
            write_bin(u_final, v_final, p_final, flag_final, imax, jmax, xlength, ylength, outfile);
        }
    }
    MPI_Finalize();

    return 0;
}

/* Save the simulation state to a file */
void write_bin(float **u, float **v, float **p, char **flag,
               int imax, int jmax, float xlength, float ylength, char *file)
{
    int i;
    FILE *fp;

    fp = fopen(file, "wb");

    if (fp == NULL)
    {
        fprintf(stderr, "Could not open file '%s': %s\n", file,
                strerror(errno));
        return;
    }

    fwrite(&imax, sizeof(int), 1, fp);
    fwrite(&jmax, sizeof(int), 1, fp);
    fwrite(&xlength, sizeof(float), 1, fp);
    fwrite(&ylength, sizeof(float), 1, fp);

    for (i = 0; i < imax + 2; i++)
    {
        fwrite(u[i], sizeof(float), jmax + 2, fp);
        fwrite(v[i], sizeof(float), jmax + 2, fp);
        fwrite(p[i], sizeof(float), jmax + 2, fp);
        fwrite(flag[i], sizeof(char), jmax + 2, fp);
    }
    fclose(fp);
}

/* Read the simulation state from a file */
int read_bin(float **u, float **v, float **p, char **flag,
             int imax, int jmax, float xlength, float ylength, char *file)
{
    int i, j;
    FILE *fp;

    if (file == NULL)
        return -1;

    if ((fp = fopen(file, "rb")) == NULL)
    {
        fprintf(stderr, "Could not open file '%s': %s\n", file,
                strerror(errno));
        fprintf(stderr, "Generating default state instead.\n");
        return -1;
    }

    fread(&i, sizeof(int), 1, fp);
    fread(&j, sizeof(int), 1, fp);
    float xl, yl;
    fread(&xl, sizeof(float), 1, fp);
    fread(&yl, sizeof(float), 1, fp);

    if (i != imax || j != jmax)
    {
        fprintf(stderr, "Warning: imax/jmax have wrong values in %s\n", file);
        fprintf(stderr, "%s's imax = %d, jmax = %d\n", file, i, j);
        fprintf(stderr, "Program's imax = %d, jmax = %d\n", imax, jmax);
        return 1;
    }
    if (xl != xlength || yl != ylength)
    {
        fprintf(stderr, "Warning: xlength/ylength have wrong values in %s\n", file);
        fprintf(stderr, "%s's xlength = %g,  ylength = %g\n", file, xl, yl);
        fprintf(stderr, "Program's xlength = %g, ylength = %g\n", xlength,
                ylength);
        return 1;
    }

    for (i = 0; i < imax + 2; i++)
    {
        fread(u[i], sizeof(float), jmax + 2, fp);
        fread(v[i], sizeof(float), jmax + 2, fp);
        fread(p[i], sizeof(float), jmax + 2, fp);
        fread(flag[i], sizeof(char), jmax + 2, fp);
    }
    fclose(fp);
    return 0;
}

/* Read the simulation state from a file */
// int mpi_read_bin(int imax, int jmax, float **u, float **v, float **p, char **flag, char *file, MPI_Datatype read_bin_type, int *sv_disp, int rank)
// {
//     int i, j;

//     MPI_FILE fh;
//     MPI_Offset offset;
//     MPI_Status status;

//     MPI_File_open(MPI_COMM_WORLD, file, MPI_MODE_RDWR, MPI_INFO_NULL, &fh);

//     int rbtsz;
//     MPI_Type_size(read_bin_type, *rbtsz);
//     offset = (MPI_Offset)(sizeof(int) * 4 + sv_disp[rank] * rbtsz);

//     MPI_File_set_view(fh, offset, read_bin_type, read_bin_type, "native", MPI_INFO_NULL);

//     for (i = 0; i < imax + 2; i++)
//     {
//         MPI_File_read_all(fh, u[i], jmax + 2, MPI_FLOAT, status);
//         MPI_File_read_all(fh, v[i], jmax + 2, MPI_FLOAT, status);
//         MPI_File_read_all(fh, p[i], jmax + 2, MPI_FLOAT, status);
//         MPI_File_read_all(fh, flag[i], jmax + 2, MPI_CHAR, status);
//     }
//     MPI_File_close(&fh);
//     return 0;
// }

static void print_usage(void)
{
    fprintf(stderr, "Try '%s --help' for more information.\n", progname);
}

static void print_version(void)
{
    fprintf(stderr, "%s %s\n", PACKAGE, VERSION);
}

static void print_help(void)
{
    fprintf(stderr, "%s. A simple computational fluid dynamics tutorial.\n\n",
            PACKAGE);
    fprintf(stderr, "Usage: %s [OPTIONS]...\n\n", progname);
    fprintf(stderr, "  -h, --help            Print a summary of the options\n");
    fprintf(stderr, "  -V, --version         Print the version number\n");
    fprintf(stderr, "  -v, --verbose=LEVEL   Set the verbosity level. 0 is silent\n");
    fprintf(stderr, "  -x, --imax=IMAX       Set the number of interior cells in the X direction\n");
    fprintf(stderr, "  -y, --jmax=JMAX       Set the number of interior cells in the Y direction\n");
    fprintf(stderr, "  -t, --t-end=TEND      Set the simulation end time\n");
    fprintf(stderr, "  -d, --del-t=DELT      Set the simulation timestep size\n");
    fprintf(stderr, "  -i, --infile=FILE     Read the initial simulation state from this file\n");
    fprintf(stderr, "                        (default is 'karman.bin')\n");
    fprintf(stderr, "  -o, --outfile=FILE    Write the final simulation state to this file\n");
    fprintf(stderr, "                        (default is 'karman.bin')\n");
}
