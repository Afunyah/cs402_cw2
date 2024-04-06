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
#include <omp.h>

void write_bin(float **u, float **v, float **p, char **flag,
               int imax, int jmax, float xlength, float ylength, char *file);

int read_bin(float **u, float **v, float **p, char **flag,
             int imax, int jmax, float xlength, float ylength, char *file);

static void print_usage(void);
static void print_version(void);
static void print_help(void);

static char *progname;

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
    double programStartTime = MPI_Wtime();

    int verbose = 1;      /* Verbosity level */
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

    int *sv_disp; // Stores offsets
    int *sv_disp2; // Stores offsets for scattering and gathering
    int *i_width_arr; // Stores widths of each node
    int *i_width_arr_exp; // Stores total number of elements for each node

    float **u_final, **v_final, **p_final;
    char **flag_final;

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_nodes);

    printf("Hello world, from process %d of %d\n", rank, n_nodes);

    delx = xlength / imax;
    dely = ylength / jmax;
    int imax_node = 0; // i workload per node

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

        // Calculate total number of elements
        for (int k = 0; k < n_nodes; k++)
        {
            i_width_arr_exp[k] = (i_width_arr[k] + 2) * (jmax + 2);
        }

        // Distibute into imax_node for each node
        MPI_Scatter(i_width_arr, 1, MPI_INT, &imax_node, 1, MPI_INT, 0, MPI_COMM_WORLD); // Send imax of each node
    }
    else
    {
        MPI_Scatter(NULL, 1, MPI_INT, &imax_node, 1, MPI_INT, 0, MPI_COMM_WORLD); // Send imax of each node
    }

    printf("%d imax node =  %d\n", rank, imax_node);

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

    if (!u || !v || !f || !g || !p || !rhs || !flag)
    {
        fprintf(stderr, "Rank %d\n", rank);
        fprintf(stderr, "Couldn't allocate memory for matrices h.\n");

        return 1;
    }

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
        // Calculate offset, which is just cumulative sum of node starting points
        int sum = 0;
        for (int i = 1; i < n_nodes; i++)
        {
            sum = sum + i_width_arr[i - 1];
            sv_disp[i] = sum;
        }
    }
    MPI_Bcast(sv_disp, n_nodes, MPI_INT, 0, MPI_COMM_WORLD);

    sv_disp2 = (int *)calloc(n_nodes, sizeof(int));
    if (rank == 0)
    {
        int sum = 0;
        for (int i = 1; i < n_nodes; i++)
        {
            sum = sum + i_width_arr[i - 1] * (jmax + 2);
            sv_disp2[i] = sum;
        }
        free(i_width_arr);
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

    // MPI_Barrier(MPI_COMM_WORLD);
    // MPI_File_close(&fh);

    float **u_full, **v_full, **p_full;
    char **flag_full;
    // Rank 0 reads the data from the file (or creates new state) and scatters data to other nodes
    if (rank == 0)
    {
        // Matrices for full grid
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

        if (init_case > 0)
        {
            /* Error while reading file */
            return 1;
        }
        if (init_case < 0)
        {
            /* Set initial values if file doesn't exist */
            // #pragma omp parallel for schedule(static) private(i,j) collapse(2)
            for (i = 0; i < imax + 2; i++)
            {
                for (j = 0; j < jmax + 2; j++)
                {
                    u_full[i][j] = ui;
                    v_full[i][j] = vi;
                    p_full[i][j] = 0.0;
                }
            }
            init_flag(flag_full, imax, jmax, delx, dely, &ibound, rank, n_nodes);
            apply_boundary_conditions(u_full, v_full, flag_full, imax, jmax, ui, vi, rank, n_nodes, 1);
        }
        if (init_case == 0)
        {
        }
    }

    // Scatter values to each node
    if (rank == 0)
    {
        MPI_Scatterv(&u_full[0][0], i_width_arr_exp, sv_disp2, MPI_FLOAT, &u[0][0], (imax_node + 2) * (jmax + 2), MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(&v_full[0][0], i_width_arr_exp, sv_disp2, MPI_FLOAT, &v[0][0], (imax_node + 2) * (jmax + 2), MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(&p_full[0][0], i_width_arr_exp, sv_disp2, MPI_FLOAT, &p[0][0], (imax_node + 2) * (jmax + 2), MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(&flag_full[0][0], i_width_arr_exp, sv_disp2, MPI_CHAR, &flag[0][0], (imax_node + 2) * (jmax + 2), MPI_CHAR, 0, MPI_COMM_WORLD);

        free_matrix(u_full);
        free_matrix(v_full);
        free_matrix(p_full);
        free_matrix(flag_full);
    }
    else
    {
        MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, &u[0][0], (imax_node + 2) * (jmax + 2), MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, &v[0][0], (imax_node + 2) * (jmax + 2), MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, &p[0][0], (imax_node + 2) * (jmax + 2), MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Scatterv(NULL, NULL, NULL, MPI_FLOAT, &flag[0][0], (imax_node + 2) * (jmax + 2), MPI_CHAR, 0, MPI_COMM_WORLD);
    }


    // Data used for alltoall
    int *disp;  // displacements

    // indicates neighbouring process to share boundaries with
    int *count_send1;
    int *count_recv1;
    int *count_send2;
    int *count_recv2;

    disp = calloc(n_nodes, sizeof(int));
    count_send1 = calloc(n_nodes, sizeof(int));
    count_recv1 = calloc(n_nodes, sizeof(int));
    count_send2 = calloc(n_nodes, sizeof(int));
    count_recv2 = calloc(n_nodes, sizeof(int));

    // Ranks should only share data with adjacent ranks
    // The following logic ensures this
    if (rank != 0)
    {
        count_send1[rank - 1] = jmax + 2;
        count_recv2[rank - 1] = jmax + 2;
    }

    if (rank != n_nodes - 1)
    {
        count_send2[rank + 1] = jmax + 2;
        count_recv1[rank + 1] = jmax + 2;
    }

    // Timers
    double timestepTime = 0.0;
    double computeVelTime = 0.0;
    double computeRhsTime = 0.0;
    double poissonTime = 0.0;
    double updateVelTime = 0.0;
    double applyBoundsTime = 0.0;

    double startTime = 0.0;
    double funcTime = 0.0;

    // Poisson not always executed
    int poisson_iters = 0;

    /* Main loop */
    for (t = 0.0; t < t_end; t += del_t, iters++)
    {
        if (rank == 0)
        {
            startTime = MPI_Wtime();
        }
        set_timestep_interval(&del_t, imax_node, jmax, delx, dely, u, v, Re, tau, rank, n_nodes);
        if (rank == 0)
        {
            funcTime = MPI_Wtime() - startTime;
            timestepTime += funcTime;
        }

        ifluid = (imax * jmax) - ibound;

        if (rank == 0)
        {
            startTime = MPI_Wtime();
        }
        compute_tentative_velocity(u, v, f, g, flag, imax_node, jmax,
                                   del_t, delx, dely, gamma, Re, rank, n_nodes, disp, count_send1, count_recv1, count_send2, count_recv2);

        if (rank == 0)
        {
            funcTime = MPI_Wtime() - startTime;
            computeVelTime += funcTime;
        }

        if (rank == 0)
        {
            startTime = MPI_Wtime();
        }
        compute_rhs(f, g, rhs, flag, imax_node, jmax, del_t, delx, dely);
        if (rank == 0)
        {
            funcTime = MPI_Wtime() - startTime;
            computeRhsTime += funcTime;
        }

        if (ifluid > 0)
        {
            if (rank == 0)
            {
                poisson_iters += 1;
                startTime = MPI_Wtime();
            }
            itersor = poisson(p, rhs, flag, imax_node, jmax, delx, dely,
                              eps, itermax, omega, &res, ifluid, rank, n_nodes, sv_disp, disp, count_send1, count_recv1, count_send2, count_recv2);
            if (rank == 0)
            {
                funcTime = MPI_Wtime() - startTime;
                poissonTime += funcTime;
            }
        }
        else
        {
            itersor = 0;
        }

        if (rank == 0 && verbose > 1)
        {
            printf("%d t:%g, del_t:%g, SOR iters:%3d, res:%e, bcells:%d\n",
                   iters, t + del_t, del_t, itersor, res, ibound);
        }

        if (rank == 0)
        {
            startTime = MPI_Wtime();
        }
        update_velocity(u, v, f, g, p, flag, imax_node, jmax, del_t, delx, dely, rank, n_nodes);
        if (rank == 0)
        {
            funcTime = MPI_Wtime() - startTime;
            updateVelTime += funcTime;
        }

        if (rank == 0)
        {
            startTime = MPI_Wtime();
        }
        apply_boundary_conditions(u, v, flag, imax_node, jmax, ui, vi, rank, n_nodes, 0);
        if (rank == 0)
        {
            funcTime = MPI_Wtime() - startTime;
            applyBoundsTime += funcTime;
        }

    } /* End of main loop */

    free(disp);
    free(count_send1);
    free(count_recv1);
    free(count_recv2);
    free(count_send2);

    if (rank == 0)
    {
        printf("\nAverage times for functions (s)\n");
        printf("------------------------------------------\n");
        printf("     set_timestep_interval : %f\n", timestepTime / (double)iters);
        printf("compute_tentative_velocity : %f\n", computeVelTime / (double)iters);
        printf("               compute_rhs : %f\n", computeRhsTime / (double)iters);
        printf("                   poisson : %f\n", poissonTime / (double)poisson_iters);
        printf("           update_velocity : %f\n", updateVelTime / (double)iters);
        printf(" apply_boundary_conditions : %f\n", applyBoundsTime / (double)iters);
        printf("\n");
    }

    // Final matrices to be written to file
    if (rank == 0)
    {
        /* Allocate arrays for full grid */
        u_final = alloc_floatmatrix(imax + 2, jmax + 2);
        v_final = alloc_floatmatrix(imax + 2, jmax + 2);
        p_final = alloc_floatmatrix(imax + 2, jmax + 2);
        flag_final = alloc_charmatrix(imax + 2, jmax + 2);
    }


    // Gather data and write to file
    if (rank == 0)
    {
        MPI_Gatherv(&u[0][0], (imax_node + 2) * (jmax + 2), MPI_FLOAT, &u_final[0][0], i_width_arr_exp, sv_disp2, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gatherv(&v[0][0], (imax_node + 2) * (jmax + 2), MPI_FLOAT, &v_final[0][0], i_width_arr_exp, sv_disp2, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gatherv(&p[0][0], (imax_node + 2) * (jmax + 2), MPI_FLOAT, &p_final[0][0], i_width_arr_exp, sv_disp2, MPI_FLOAT, 0, MPI_COMM_WORLD);
        MPI_Gatherv(&flag[0][0], (imax_node + 2) * (jmax + 2), MPI_CHAR, &flag_final[0][0], i_width_arr_exp, sv_disp2, MPI_CHAR, 0, MPI_COMM_WORLD);

        free(i_width_arr_exp);
    }
    else
    {
        MPI_Gatherv(&u[0][0], (imax_node + 2) * (jmax + 2), MPI_FLOAT, NULL, NULL, NULL, NULL, 0, MPI_COMM_WORLD);
        MPI_Gatherv(&v[0][0], (imax_node + 2) * (jmax + 2), MPI_FLOAT, NULL, NULL, NULL, NULL, 0, MPI_COMM_WORLD);
        MPI_Gatherv(&p[0][0], (imax_node + 2) * (jmax + 2), MPI_FLOAT, NULL, NULL, NULL, NULL, 0, MPI_COMM_WORLD);
        MPI_Gatherv(&flag[0][0], (imax_node + 2) * (jmax + 2), MPI_CHAR, NULL, NULL, NULL, NULL, 0, MPI_COMM_WORLD);
    }
    free(sv_disp2);

    if (rank == 0)
    {
        if (outfile != NULL && strcmp(outfile, "") != 0)
        {
            write_bin(u_final, v_final, p_final, flag_final, imax, jmax, xlength, ylength, outfile);
        }

        free_matrix(u_final);
        free_matrix(v_final);
        free_matrix(p_final);
        free_matrix(flag_final);
    }

    free_matrix(u);
    free_matrix(v);
    free_matrix(f);
    free_matrix(g);
    free_matrix(p);
    free_matrix(rhs);
    free_matrix(flag);

    if (rank == 0)
    {
        printf("RUNTIME: %f\n\n", MPI_Wtime() - programStartTime);
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
