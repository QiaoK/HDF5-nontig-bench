#include <stdio.h>
#include <stdlib.h>
#include <string.h> /* strcpy(), strncpy() */
#include <unistd.h> /* getopt() */
#include "hdf5.h"

#define ENABLE_MULTIDATASET 0
#define MULTIDATASET_DEFINE 1

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

static int flush_multidatasets() {
    int i;
    uint32_t local_no_collective_cause, global_no_collective_cause;
    int rank;
    size_t esize;
    hsize_t dims[H5S_MAX_RANK], mdims[H5S_MAX_RANK];
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

    for ( i = 0; i < dataset_size; ++i ) {
        H5Sget_simple_extent_dims (multi_datasets[i].mem_space_id, dims, mdims);
        esize = H5Tget_size (multi_datasets[i].mem_type_id);
    }

    //printf("rank %d number of hyperslab called %d\n", rank, hyperslab_count);

    if (dataset_size) {
        free(multi_datasets);
    }
    dataset_size = 0;
    dataset_size_limit = 0;
    return 0;
}

int fill_data_buffer(char*** buf, int n_datasets, int rank, int ndim, hsize_t *dims) {
    int i;
    hsize_t total_data_size = 1, j;
    for ( i = 0; i < ndim; ++i ) {
        total_data_size *= dims[i];
    }

    buf[0] = (char**) malloc(sizeof(char*) * n_datasets);
    for ( i = 0; i < n_datasets; ++i ) {
        buf[0][i] = (char*) malloc(sizeof(char) * total_data_size);
        for ( j = 0; j < total_data_size; ++j ) {
            buf[0][i][j] = rank + i * 13 + j;
        }
    }
    return 0;
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

int close_datasets(hid_t *dids, int n_datasets) {
    int i;
    for ( i = 0; i < n_datasets; ++i ) {
        H5Dclose(dids[i]);
    }
    return 0;
}

int aggregate_datasets(hid_t did, char* buf, int req_count, int req_size, int ndim, hsize_t *dims, int rank, int nprocs) {
    int i, j;
    hid_t dsid, msid;
    hsize_t start[H5S_MAX_RANK], block[H5S_MAX_RANK];
    hsize_t total_memspace_size = 1;

    dsid = H5Dget_space (did);
    register_dataspace_recycle(dsid);

    if (ndim == 1) {
        for ( i = 0; i < req_count; ++i ) {
            start[0] = i * nprocs * req_size + req_size * rank;
            block[0] = req_size;
            if ( i ) {
                H5Sselect_hyperslab (dsid, H5S_SELECT_OR, start, NULL, one, block);
            } else {
                H5Sselect_hyperslab (dsid, H5S_SELECT_SET, start, NULL, one, block);
            }
        }
        total_memspace_size *= req_size * req_count;
    }
    msid = H5Screate_simple (1, &total_memspace_size, &total_memspace_size);
    register_memspace_recycle(msid);
    register_multidataset(buf, did, dsid, msid, H5T_NATIVE_CHAR, 1);
    return 0;
}

int main (int argc, char **argv) {
    int i, ndim, n_datasets, req_count, rank, nprocs;
    size_t req_size = 0;
    hsize_t *dims;
    hid_t faplid, fid, *dids;
    char **buf;
    char outfname[128];

    sprintf(outfname, "test.h5");

    memspace_recycle_size = 0;
    memspace_recycle_size_limit = 0;

    dataspace_recycle_size = 0;
    dataspace_recycle_size_limit = 0;

    dataset_size = 0;
    dataset_size_limit = 0;

    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &nprocs);

    while ((i = getopt (argc, argv, "d:r:n:c:")) != EOF) switch (i) {
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
    }

    for (i = 0; i < H5S_MAX_RANK; i++) {
        one[i]  = 1;
    }

    faplid = H5Pcreate (H5P_FILE_ACCESS);
    H5Pset_fapl_mpio (faplid, MPI_COMM_WORLD, MPI_INFO_NULL);

    fid = H5Fcreate (outfname, H5F_ACC_TRUNC, H5P_DEFAULT, faplid);

    dims = (hsize_t*) malloc(sizeof(hsize_t) * ndim);
    for ( i = 0; i < ndim; ++i ) {
        dims[i] = req_count * req_size * nprocs;
    }

    create_datasets(fid, &dids, n_datasets, ndim, dims);

    fill_data_buffer(&buf, n_datasets, rank, ndim, dims);

    for ( i = 0; i < n_datasets; ++i ) {
        aggregate_datasets(dids[i], buf[i], req_count, req_size, ndim, dims, rank, nprocs);
    }
    free(dims);

    H5Fclose(fid);

    MPI_Finalize ();
    return 0;
}
