#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strcpy(), strncpy() */
#include <unistd.h> /* getopt() */
#include <math.h>
#include <stdint.h>
#include "hdf5.h"
#include "random.h"

#define ENABLE_MULTIDATASET 0
#define MULTIDATASET_DEFINE 0

#if MULTIDATASET_DEFINE == 1
typedef struct H5D_rw_multi_t
{
    hid_t dset_id;          /* dataset ID */
    hid_t dset_space_id;    /* dataset selection dataspace ID */
    hid_t mem_type_id;      /* memory datatype ID */
    hid_t mem_space_id;     /* memory selection dataspace ID */
    union {
        void *rbuf;         /* pointer to read buffer */
        const void *wbuf;   /* pointer to write buffer */
    } u;
} H5D_rw_multi_t;
#endif

typedef struct hdf5_noncontig_timing {
    double file_create;
    double dataset_create;
    double dataset_hyperslab;
    double dataset_io;
    double dataset_close;
    double file_close;
} hdf5_noncontig_timing;

static int dataset_size;
static int dataset_size_limit;
static int dataspace_recycle_size;
static int dataspace_recycle_size_limit;
static int memspace_recycle_size;
static int memspace_recycle_size_limit;

static hsize_t one[H5S_MAX_RANK];

static H5D_rw_multi_t *multi_datasets;
static hid_t *memspace_recycle;
static hid_t *dataspace_recycle;

static int dataspace_recycle_all() {
    int i;
    //printf("recycle %d dataspace\n", dataspace_recycle_size);
    for ( i = 0; i < dataspace_recycle_size; ++i ) {
        if ( dataspace_recycle[i] >= 0 ) {
            H5Sclose(dataspace_recycle[i]);
        }
    }
    if (dataspace_recycle_size) {
        free(dataspace_recycle);
    }
    dataspace_recycle_size = 0;
    dataspace_recycle_size_limit = 0;
    return 0;
}

static int memspace_recycle_all() {
    int i;
    //printf("recycle %d memspace\n", memspace_recycle_size);
    for ( i = 0; i < memspace_recycle_size; ++i ) {
        if ( memspace_recycle[i] >= 0 ){
            H5Sclose(memspace_recycle[i]);
        }
    }
    if (memspace_recycle_size) {
        free(memspace_recycle);
    }
    memspace_recycle_size = 0;
    memspace_recycle_size_limit = 0;
    return 0;
}

static int recycle_all() {
    dataspace_recycle_all();
    memspace_recycle_all();
    return 0;
}


static int register_dataspace_recycle(hid_t dsid) {
    if (dataspace_recycle_size == dataspace_recycle_size_limit) {
        if ( dataspace_recycle_size_limit > 0 ) {
            dataspace_recycle_size_limit *= 2;
            hid_t *temp = (hid_t*) malloc(dataspace_recycle_size_limit*sizeof(hid_t));
            memcpy(temp, dataspace_recycle, sizeof(hid_t) * dataspace_recycle_size);
            free(dataspace_recycle);
            dataspace_recycle = temp;
        } else {
            dataspace_recycle_size_limit = 512;
            dataspace_recycle = (hid_t*) malloc(dataspace_recycle_size_limit*sizeof(hid_t));
        }
    }
    dataspace_recycle[dataspace_recycle_size] = dsid;
    dataspace_recycle_size++;
    return 0;
}

static int register_memspace_recycle(hid_t msid) {
    if (memspace_recycle_size == memspace_recycle_size_limit) {
        if ( memspace_recycle_size_limit > 0 ) {
            memspace_recycle_size_limit *= 2;
            hid_t *temp = (hid_t*) malloc(memspace_recycle_size_limit*sizeof(hid_t));
            memcpy(temp, memspace_recycle, sizeof(hid_t) * memspace_recycle_size);
            free(memspace_recycle);
            memspace_recycle = temp;
        } else {
            memspace_recycle_size_limit = 512;
            memspace_recycle = (hid_t*) malloc(memspace_recycle_size_limit*sizeof(hid_t));
        }
    }
    memspace_recycle[memspace_recycle_size] = msid;
    memspace_recycle_size++;
    return 0;
}

static int register_multidataset(void *buf, hid_t did, hid_t dsid, hid_t msid, hid_t mtype, int write) {
    if (dataset_size == dataset_size_limit) {
        if ( dataset_size_limit > 0 ) {
            dataset_size_limit *= 2;
            H5D_rw_multi_t *temp = (H5D_rw_multi_t*) malloc(dataset_size_limit*sizeof(H5D_rw_multi_t));
            memcpy(temp, multi_datasets, sizeof(H5D_rw_multi_t) * dataset_size);
            free(multi_datasets);
            multi_datasets = temp;
        } else {
            dataset_size_limit = 512;
            multi_datasets = (H5D_rw_multi_t*) malloc(dataset_size_limit*sizeof(H5D_rw_multi_t));
        }
    }

    multi_datasets[dataset_size].mem_space_id = msid;
    multi_datasets[dataset_size].dset_id = did;
    multi_datasets[dataset_size].dset_space_id = dsid;
    multi_datasets[dataset_size].mem_type_id = mtype;
    if (write) {
        multi_datasets[dataset_size].u.wbuf = buf;
    } else {
        multi_datasets[dataset_size].u.rbuf = buf;
    }
    dataset_size++;
    return 0;
}

int print_no_collective_cause(uint32_t local_no_collective_cause,uint32_t global_no_collective_cause) {
    switch (local_no_collective_cause) {
    case H5D_MPIO_COLLECTIVE: {
        //printf("MPI-IO collective successful\n");
        break;
    }
    case H5D_MPIO_SET_INDEPENDENT: {
        printf("local flag: MPI-IO independent flag is on\n");
        break;
    }
    case H5D_MPIO_DATATYPE_CONVERSION  : {
        printf("local flag: MPI-IO datatype conversion needed\n");
        break;
    }
    case H5D_MPIO_DATA_TRANSFORMS: {
        printf("local flag: MPI-IO H5D_MPIO_DATA_TRANSFORMS.\n");
        break;
    }
/*
    case H5D_MPIO_SET_MPIPOSIX: {
        printf("local flag: MPI-IO H5D_MPIO_SET_MPIPOSIX \n");
    }
*/
    case H5D_MPIO_NOT_SIMPLE_OR_SCALAR_DATASPACES: {
        printf("local flag: MPI-IO NOT_SIMPLE_OR_SCALAR_DATASPACES\n");
        break;
    }
/*
    case H5D_MPIO_POINT_SELECTIONS: {
        printf("local flag: MPI-IO H5D_MPIO_POINT_SELECTIONS\n");
    }
*/
    case H5D_MPIO_NOT_CONTIGUOUS_OR_CHUNKED_DATASET: {
        printf("local flag: MPI-IO H5D_MPIO_NOT_CONTIGUOUS_OR_CHUNKED_DATASET\n");
        break;
    }
/*
    case H5D_MPIO_FILTERS: {
        printf("local flag: MPI-IO H5D_MPIO_FILTERS\n");
        break;
    }
*/
    default: {
        printf("undefined label for collective cause\n");
        break;
    }
    }

    switch (global_no_collective_cause) {
    case H5D_MPIO_COLLECTIVE: {
        //printf("MPI-IO collective successful\n");
        break;
    }
    case H5D_MPIO_SET_INDEPENDENT: {
        printf("global flag: MPI-IO independent flag is on\n");
        break;
    }
    case H5D_MPIO_DATATYPE_CONVERSION  : {
        printf("global flag: MPI-IO datatype conversion needed\n");
        break;
    }
    case H5D_MPIO_DATA_TRANSFORMS: {
        printf("global flag: MPI-IO H5D_MPIO_DATA_TRANSFORMS.\n");
        break;
    }
/*
    case H5D_MPIO_SET_MPIPOSIX: {
        printf("global flag: MPI-IO H5D_MPIO_SET_MPIPOSIX \n");
    }
*/
    case H5D_MPIO_NOT_SIMPLE_OR_SCALAR_DATASPACES: {
        printf("global flag: MPI-IO NOT_SIMPLE_OR_SCALAR_DATASPACES\n");
        break;
    }
/*
    case H5D_MPIO_POINT_SELECTIONS: {
        printf("global flag: MPI-IO H5D_MPIO_POINT_SELECTIONS\n");
    }
*/
    case H5D_MPIO_NOT_CONTIGUOUS_OR_CHUNKED_DATASET: {
        printf("global flag: MPI-IO H5D_MPIO_NOT_CONTIGUOUS_OR_CHUNKED_DATASET\n");
        break;
    }
/*
    case H5D_MPIO_FILTERS: {
        printf("global flag: MPI-IO H5D_MPIO_FILTERS\n");
        break;
    }
*/
    default: {
        printf("undefined label for collective cause\n");
        break;
    }
    }
    return 0;
}

static int pull_multidatasets() {
    int i;
    uint32_t local_no_collective_cause, global_no_collective_cause;
    int rank;
    hid_t dxplid_coll = H5Pcreate (H5P_DATASET_XFER);
    H5Pset_dxpl_mpio (dxplid_coll, H5FD_MPIO_COLLECTIVE);

    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    //printf("Rank %d number of datasets to be written %d\n", rank, dataset_size);
#if ENABLE_MULTIDATASET==1
    H5Dread_multi(dxplid_coll, dataset_size, multi_datasets);
#else

    //printf("rank %d has dataset_size %lld\n", rank, (long long int) dataset_size);
    for ( i = 0; i < dataset_size; ++i ) {
        //MPI_Barrier(MPI_COMM_WORLD);
        H5Dread (multi_datasets[i].dset_id, multi_datasets[i].mem_type_id, multi_datasets[i].mem_space_id, multi_datasets[i].dset_space_id, dxplid_coll, multi_datasets[i].u.rbuf);
        if (!rank) {
            H5Pget_mpio_no_collective_cause( dxplid_coll, &local_no_collective_cause, &global_no_collective_cause);
            print_no_collective_cause(local_no_collective_cause, global_no_collective_cause);
        }

    }
#endif

    if (dataset_size) {
        free(multi_datasets);
    }
    dataset_size = 0;
    dataset_size_limit = 0;
    return 0;
}

static int flush_multidatasets() {
    int i;
    uint32_t local_no_collective_cause, global_no_collective_cause;
    int rank;
    hid_t dxplid_coll = H5Pcreate (H5P_DATASET_XFER);
    H5Pset_dxpl_mpio (dxplid_coll, H5FD_MPIO_COLLECTIVE);

    MPI_Comm_rank (MPI_COMM_WORLD, &rank);

    //printf("Rank %d number of datasets to be written %d\n", rank, dataset_size);
#if ENABLE_MULTIDATASET==1
    H5Dwrite_multi(dxplid_coll, dataset_size, multi_datasets);
#else

    //printf("rank %d has dataset_size %lld\n", rank, (long long int) dataset_size);
    for ( i = 0; i < dataset_size; ++i ) {
        //MPI_Barrier(MPI_COMM_WORLD);
        H5Dwrite (multi_datasets[i].dset_id, multi_datasets[i].mem_type_id, multi_datasets[i].mem_space_id, multi_datasets[i].dset_space_id, dxplid_coll, multi_datasets[i].u.wbuf);

        if (!rank) {
            H5Pget_mpio_no_collective_cause( dxplid_coll, &local_no_collective_cause, &global_no_collective_cause);
            print_no_collective_cause(local_no_collective_cause, global_no_collective_cause);
        }

    }
#endif

    if (dataset_size) {
        free(multi_datasets);
    }
    dataset_size = 0;
    dataset_size_limit = 0;
    return 0;
}

int fill_data_buffer(char*** buf, int n_datasets, int rank, hsize_t total_data_size, int data_init_flag) {
    int i;
    hsize_t j;

    buf[0] = (char**) malloc(sizeof(char*) * n_datasets);
    for ( i = 0; i < n_datasets; ++i ) {
        buf[0][i] = (char*) malloc(sizeof(char) * total_data_size);
        if (data_init_flag) {
            for ( j = 0; j < total_data_size; ++j ) {
                buf[0][i][j] = rank + i * 13 + j;
            }
        }
    }
    return 0;
}

int compare_data_buffer(char **buf1, char **buf2, hsize_t total_data_size) {
    hsize_t j;
    for ( j = 0; j < total_data_size; ++j ) {
        if (buf1[0][j] != buf2[0][j]) {
            return 0;
        }
    }
    return 1;
}

int free_data_buffer(char** buf, int n_datasets) {
    int i;
    for ( i = 0; i < n_datasets; ++i ) {
        free(buf[i]);
    }
    free(buf);
    return 0;
}

int create_datasets(hid_t fid, hid_t **dids, int n_datasets, int ndim, hsize_t *dims) {
    int i;
    hid_t dcplid = -1;
    hid_t sid;
    char name[128];

    dcplid = H5Pcreate (H5P_DATASET_CREATE);
    H5Pset_fill_value(dcplid, 0, NULL );
    H5Pset_fill_time(dcplid, H5D_FILL_TIME_NEVER);
    H5Pset_alloc_time(dcplid, H5D_ALLOC_TIME_DEFAULT );

    dids[0] = (hid_t*) malloc(sizeof(hid_t) * n_datasets);
    sid = H5Screate_simple (ndim, dims, dims);
    for ( i = 0; i < n_datasets; ++i ) {
        sprintf(name, "dataset_%d", i);
        dids[0][i] = H5Dcreate2 (fid, name, H5T_NATIVE_CHAR, sid, H5P_DEFAULT, dcplid, H5P_DEFAULT);
    }
    H5Pclose(dcplid);
    H5Sclose(sid);
    return 0;
}

int open_datasets(hid_t fid, hid_t **dids, int n_datasets) {
    int i;
    char name[128];

    dids[0] = (hid_t*) malloc(sizeof(hid_t) * n_datasets);
    for ( i = 0; i < n_datasets; ++i ) {
        sprintf(name, "dataset_%d", i);
        dids[0][i] = H5Dopen2 (fid, name, H5P_DEFAULT);
    }
    return 0;
}

int close_datasets(hid_t *dids, int n_datasets) {
    int i;
    for ( i = 0; i < n_datasets; ++i ) {
        H5Dclose(dids[i]);
    }
    return 0;
}

int aggregate_datasets(hid_t did, char* buf, int req_count, int req_size, int ndim, const hsize_t *dims, const hsize_t *req_offset) {
    int i;
    hid_t dsid, msid;
    hsize_t start[H5S_MAX_RANK], block[H5S_MAX_RANK];
    hsize_t total_memspace_size = 0;

    dsid = H5Dget_space (did);
    register_dataspace_recycle(dsid);

    if (ndim == 1) {
        for ( i = 0; i < req_count; ++i ) {
            start[0] = req_offset[i];
            block[0] = req_size;
            total_memspace_size += block[0];
            if ( i ) {
                H5Sselect_hyperslab (dsid, H5S_SELECT_OR, start, NULL, one, block);
            } else {
                H5Sselect_hyperslab (dsid, H5S_SELECT_SET, start, NULL, one, block);
            }
        }
    } else if (ndim == 2) {
        for ( i = 0; i < req_count; ++i ) {
            start[0] = (req_offset[i] / (dims[1]/req_size) ) * req_size;
            start[1] = (req_offset[i] % (dims[1]/req_size) ) * req_size;
            block[0] = req_size;
            block[1] = req_size;
            total_memspace_size += block[0] * block[1];
            if ( i ) {
                H5Sselect_hyperslab (dsid, H5S_SELECT_OR, start, NULL, one, block);
            } else {
                H5Sselect_hyperslab (dsid, H5S_SELECT_SET, start, NULL, one, block);
            }
        }
    } else if (ndim == 3) {
        for ( i = 0; i < req_count; ++i ) {
            start[0] = (req_offset[i] / ( (dims[1]/req_size) * (dims[2]/req_size) )) * req_size;
            start[1] = (( req_offset[i] % ( (dims[2]/req_size) * (dims[1]/req_size) )) / (dims[2]/req_size) ) * req_size;
            start[2] = (req_offset[i] % (dims[2]/req_size)) * req_size;
            block[0] = req_size;
            block[1] = req_size;
            block[2] = req_size;
            //printf("start[0] = %llu, start[1] = %llu, start[2] = %llu, block[0] = %llu, block[1] = %llu, block[2] = %llu\n", start[0], start[1], start[2], block[0], block[1], block[2]);
            total_memspace_size += block[0] * block[1] * block[2];
            if ( i ) {
                H5Sselect_hyperslab (dsid, H5S_SELECT_OR, start, NULL, one, block);
            } else {
                H5Sselect_hyperslab (dsid, H5S_SELECT_SET, start, NULL, one, block);
            }
        }
    }
    msid = H5Screate_simple (1, &total_memspace_size, &total_memspace_size);
    register_memspace_recycle(msid);
    register_multidataset(buf, did, dsid, msid, H5T_NATIVE_CHAR, 1);
    return 0;
}

int report_timings(hdf5_noncontig_timing *timings, int rank, const char *prefix, int nprocs, uint64_t local_data_size) {
    hdf5_noncontig_timing max_times;
    uint64_t total_data_size;

    MPI_Reduce(timings, &max_times, sizeof(hdf5_noncontig_timing) / sizeof(double), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_data_size, &total_data_size, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Total dataset written by %d processes is %llu MB\n", nprocs, (long long unsigned) (total_data_size / 1048576));
        printf("%s file create   : %lf (%lf) seconds\n", prefix, timings->file_create, max_times.file_create);
        printf("%s dataset create: %lf (%lf) seconds\n", prefix, timings->dataset_create, max_times.dataset_create);
        printf("%s dataset hyperslab: %lf (%lf) seconds\n", prefix, timings->dataset_hyperslab, max_times.dataset_hyperslab);
        printf("%s dataset I/O : %lf (%lf) seconds\n", prefix, timings->dataset_io, max_times.dataset_io);
        printf("%s dataset close : %lf (%lf) seconds\n", prefix, timings->dataset_close, max_times.dataset_close);
        printf("%s file close    : %lf (%lf) seconds\n", prefix, timings->file_close, max_times.file_close);
    }

    return 0;
}

int set_dataset_dimensions(int rank, int nprocs, int ndim, hsize_t *dims, int req_count, int req_size) {
    int req_count_per_dim;
    if (ndim == 1) {
        dims[0] = req_count * nprocs * req_size;
        if ( rank == 0 ) {
            printf("ndim = %d, dims[0] = %llu\n", ndim, dims[0]);
        }
    } else if (ndim == 2) {
        req_count_per_dim = ((int) ceil(sqrt(nprocs * req_count))) * req_size;
        dims[0] = req_count_per_dim;
        dims[1] = req_count_per_dim;
        if ( rank == 0 ) {
            printf("ndim = %d, dims[0] = %llu, dims[1] = %llu\n", ndim, dims[0], dims[1]);
        }
    } else if (ndim ==3) {
        req_count_per_dim = ((int) ceil(cbrt(req_count * nprocs))) * req_size;
        dims[0] = req_count_per_dim;
        dims[1] = req_count_per_dim;
        dims[2] = req_count_per_dim;
        if ( rank == 0 ) {
            printf("ndim = %d, dims[0] = %llu, dims[1] = %llu, dims[2] = %llu\n", ndim, dims[0], dims[1], dims[2]);
        }
    }
    return 0;
}

void shuffle(hsize_t *array, hsize_t n)
{
    if (n > 1) 
    {
        hsize_t i;
        for (i = 1; i < n ; i++) 
        {
          hsize_t j = genrand_int32() % ( n - i + 1);
          hsize_t t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}

int initialize_requests(int rank, int nprocs, int type, int req_count, int n_datasets, hsize_t ***req_offset) {
    int i, j;
    hsize_t *random_array = NULL;
    *req_offset = (hsize_t**) malloc(sizeof(hsize_t*) * n_datasets);
    **req_offset = (hsize_t*) malloc(sizeof(hsize_t) * n_datasets * req_count);
    for ( i = 1; i < n_datasets; ++i ) {
        req_offset[0][i] = req_offset[0][i-1] + req_count;
    }

    switch (type) {
        case 0: {
            for ( j = 0; j < n_datasets; ++j ) {
                for ( i = 0; i < req_count; ++i ) {
                    req_offset[0][j][i] = (i * nprocs + rank);
                }
            }
            break;
        }
        case 1: {
            if (rank ==0) {
                for ( j = 0; j < n_datasets; ++j ) {
                    random_array = (hsize_t*) malloc(sizeof(hsize_t) * nprocs * req_count);
                    for ( i = 0; i < nprocs * req_count; ++i ) {
                        random_array[i] = i;
                    }
                    shuffle(random_array, nprocs * req_count);
                    MPI_Scatter(random_array, req_count * sizeof(hsize_t), MPI_BYTE, req_offset[0][j], req_count * sizeof(hsize_t), MPI_BYTE, 0, MPI_COMM_WORLD);
                }
            } else {
                for ( j = 0; j < n_datasets; ++j ) {
                    MPI_Scatter(NULL, 0, MPI_BYTE, req_offset[0][j], req_count * sizeof(hsize_t), MPI_BYTE, 0, MPI_COMM_WORLD);
                }
            }
            break;
        }
    }
    return 0;
}

int finalize_requests(hsize_t **req_offset) {
    free(req_offset[0]);
    free(req_offset);
    return 0;
}

int process_read(int rank, int nprocs, int n_datasets, int ndim, int req_count, size_t req_size, hsize_t **req_offset, const char *outfname, char ***buf_ptr) {
    int i;
    char **buf;
    hsize_t *dims;
    hid_t faplid, fid, *dids;
    double start;
    hdf5_noncontig_timing *timings;
    hsize_t total_data_size;

    timings = calloc(1, sizeof(hdf5_noncontig_timing));

    start = MPI_Wtime();
    faplid = H5Pcreate (H5P_FILE_ACCESS);
    H5Pset_fapl_mpio (faplid, MPI_COMM_WORLD, MPI_INFO_NULL);

    fid = H5Fopen (outfname, H5F_ACC_RDONLY, faplid);
    timings->file_create = MPI_Wtime() - start;

    dims = (hsize_t*) malloc(sizeof(hsize_t) * ndim);
    set_dataset_dimensions(rank, nprocs, ndim, dims, req_count, req_size);
    start = MPI_Wtime();
    open_datasets(fid, &dids, n_datasets);
    timings->dataset_create = MPI_Wtime() - start;

    total_data_size = req_count * req_size;
    for ( i = 1; i < ndim; ++i ) {
        total_data_size *= req_size;
    }
    fill_data_buffer(&buf, n_datasets, rank, total_data_size, 0);

    start = MPI_Wtime();
    for ( i = 0; i < n_datasets; ++i ) {
        aggregate_datasets(dids[i], buf[i], req_count, req_size, ndim, dims, req_offset[i]);
    }
    timings->dataset_hyperslab = MPI_Wtime() - start;

    start = MPI_Wtime();
    pull_multidatasets();
    timings->dataset_io = MPI_Wtime() - start;

    free_data_buffer(buf, n_datasets);
    recycle_all();

    start = MPI_Wtime();
    close_datasets(dids, n_datasets);
    timings->dataset_close = MPI_Wtime() - start;

    start = MPI_Wtime();
    H5Fclose(fid);
    H5Pclose(faplid);
    timings->file_close = MPI_Wtime() - start;

    report_timings(timings, rank, "HDF5 read", nprocs, n_datasets * total_data_size);

    free(dims);
    free(timings);

    *buf_ptr = buf;
    return 0;
}

int process_write(int rank, int nprocs, int n_datasets, int ndim, int req_count, size_t req_size, hsize_t **req_offset, const char *outfname, char ***buf_ptr, hsize_t *total_data_size_ptr) {
    int i;
    char **buf;
    hsize_t *dims;
    hid_t faplid, fid, *dids;
    double start;
    hdf5_noncontig_timing *timings;
    hsize_t total_data_size;

    timings = calloc(1, sizeof(hdf5_noncontig_timing));

    start = MPI_Wtime();
    faplid = H5Pcreate (H5P_FILE_ACCESS);
    H5Pset_fapl_mpio (faplid, MPI_COMM_WORLD, MPI_INFO_NULL);

    fid = H5Fcreate (outfname, H5F_ACC_TRUNC, H5P_DEFAULT, faplid);
    timings->file_create = MPI_Wtime() - start;

    dims = (hsize_t*) malloc(sizeof(hsize_t) * ndim);
    set_dataset_dimensions(rank, nprocs, ndim, dims, req_count, req_size);
    start = MPI_Wtime();
    create_datasets(fid, &dids, n_datasets, ndim, dims);
    timings->dataset_create = MPI_Wtime() - start;

    total_data_size = req_count * req_size;
    for ( i = 1; i < ndim; ++i ) {
        total_data_size *= req_size;
    }
    fill_data_buffer(&buf, n_datasets, rank, total_data_size, 1);

    start = MPI_Wtime();
    for ( i = 0; i < n_datasets; ++i ) {
        aggregate_datasets(dids[i], buf[i], req_count, req_size, ndim, dims, req_offset[i]);
    }
    timings->dataset_hyperslab = MPI_Wtime() - start;
    //printf("checkpoint %d\n", rank);
    start = MPI_Wtime();
    flush_multidatasets();
    timings->dataset_io = MPI_Wtime() - start;

    recycle_all();

    start = MPI_Wtime();
    close_datasets(dids, n_datasets);
    timings->dataset_close = MPI_Wtime() - start;

    start = MPI_Wtime();
    H5Fclose(fid);
    H5Pclose(faplid);
    timings->file_close = MPI_Wtime() - start;

    report_timings(timings, rank, "HDF5 write", nprocs, n_datasets * total_data_size);

    free(dims);
    free(timings);

    *buf_ptr = buf;
    *total_data_size_ptr = total_data_size;
    return 0;
}

int main (int argc, char **argv) {
    int i, ndim = 1, n_datasets = 1, req_count = 0, rank, nprocs;
    size_t req_size = 0;
    int req_type = 0;
    int read_flag = 0, write_flag = 0;
    char filename[256];
    char **buf1, char **buf2;
    hsize_t **req_offset;
    hsize_t total_data_size;
    int compare_correctness = 0;

    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);

    strcpy(filename, "test.h5");

    init_genrand(5555);

    memspace_recycle_size = 0;
    memspace_recycle_size_limit = 0;

    dataspace_recycle_size = 0;
    dataspace_recycle_size_limit = 0;

    dataset_size = 0;
    dataset_size_limit = 0;

    while ((i = getopt (argc, argv, "CWRt:d:s:n:c:a:")) != EOF) switch (i) {
        case 'a': {
            strcpy(filename, optarg);
            break;
        }
        case 'c': {
            req_count = atoi(optarg);
            break;
        }
        case 's': {
            req_size = atoi(optarg);
            break;
        }
        case 'n': {
            ndim = atoi(optarg);
            break;
        }
        case 'd': {
            n_datasets = atoi(optarg);
            break;
        }
        case 't': {
            req_type = atoi(optarg);
            break;
        }
        case 'R': {
            read_flag = 1;
            break;
        }
        case 'W': {
            write_flag = 1;
            break;
        }
        case 'C': {
            compare_correctness = 1;
            break;
        }
        default: {
            if (rank == 0) printf("arguments are insufficient\n");
            MPI_Finalize ();
            return 1;
        }
    }
    if (rank == 0) {
        printf("req_count = %d, req_size = %ld, ndim = %d, n_datasets = %d, req_type = %d\n", req_count, req_size, ndim, n_datasets, req_type);
    }
    for (i = 0; i < H5S_MAX_RANK; i++) {
        one[i]  = 1;
    }
    initialize_requests(rank, nprocs, req_type, req_count, n_datasets, &req_offset);
    if (write_flag) {
        process_write(rank, nprocs, n_datasets, ndim, req_count, req_size, req_offset, filename, &buf1, &total_data_size);
    }
    if (read_flag) {
        process_read(rank, nprocs, n_datasets, ndim, req_count, req_size, req_offset, filename, &buf2);
    }
    if (compare_correctness && read_flag && write_flag) {
        if (compare_data_buffer(buf1, buf2, total_data_size)) {
            printf("Byte-wise correctness check passed\n");
        } else {
            printf("Byte-wise correctness check failed\n");
        }
    }

    free_data_buffer(buf1, n_datasets);
    free_data_buffer(buf2, n_datasets);
    finalize_requests(req_offset);
    MPI_Finalize ();
    return 0;
}
