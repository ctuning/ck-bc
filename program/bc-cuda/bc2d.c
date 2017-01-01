#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdarg.h>
#include <getopt.h>
#include <cuda_runtime.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dlfcn.h>
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include "bc2d.h"



#define _TIMINGS 1
#define BIN_NUM 16

#define LIMIT 1
uint64_t N=0;   /* number of vertices: N */
LOCINT  row_bl; /* adjacency matrix rows per block: N/(RC) */
LOCINT  col_bl; /* adjacency matrix columns per block: N/C */
LOCINT  row_pp; /* adjacency matrix rows per proc: N/(RC) * C = N/R */
uint64_t degree_reduction_time = 0;
uint64_t overlap_time = 0;
uint64_t two_degree_reduction_time = 0;
uint64_t sort_time = 0;
int C=0;
int R=0;
int gmyid;
int myid;
int gntask;
int ntask;
int mono = 1;
int undirected = 1;
LOCINT *tlvl = NULL;
LOCINT *tlvl_v1 = NULL;

int heuristic=0;

int myrow;
int mycol;
int pmesh[MAX_PROC_I][MAX_PROC_J];

LOCINT flag = 0;

LOCINT *reach=NULL;
FILE * outdebug = NULL;
LOCINT loc_count = 0;
STATDATA *mystats = NULL;
unsigned int outId;
char strmesh[10];
char cmdLine[256];


static void freeMem (void* p) {
	if (p) free(p);
}

static void prexit(const char *fmt, ...) {

	int myid=0;
	va_list ap;

	va_start(ap, fmt);
	if (0 == myid) vfprintf(stderr, fmt, ap);
	exit(EXIT_FAILURE);
}

void *Malloc(size_t sz) {

	void *ptr;

	ptr = (void *) malloc(sz);
	if (!ptr) {
		fprintf(stderr, "Cannot allocate %zu bytes...\n", sz);
		exit(EXIT_FAILURE);
	}
	memset(ptr, 0, sz);
	return ptr;
}




static void *Realloc(void *ptr, size_t sz) {

	void *lp;

	lp = (void *) realloc(ptr, sz);
	if (!lp && sz) {
		fprintf(stderr, "Cannot reallocate to %zu bytes...\n", sz);
		exit(EXIT_FAILURE);
	}
	return lp;
}

static FILE *Fopen(const char *path, const char *mode) {

	FILE *fp = NULL;
	fp = fopen(path, mode);
	if (!fp) {
		fprintf(stderr, "Cannot open file %s...\n", path);
		exit(EXIT_FAILURE);
	}
	return fp;
}

static off_t get_fsize(const char *fpath) {

	struct stat st;
	int rv;

	rv = stat(fpath, &st);
	if (rv) {
		fprintf(stderr, "Cannot stat file %s...\n", fpath);
		exit(EXIT_FAILURE);
	}
	return st.st_size;
}

static uint64_t getFsize(FILE *fp) {

	int rv;
	uint64_t size = 0;

	rv = fseek(fp, 0, SEEK_END);
	if (rv != 0) {
		fprintf(stderr, "SEEK END FAILED\n");
		if (ferror(fp)) fprintf(stderr, "FERROR SET\n");
		exit(EXIT_FAILURE);
	}

	size = ftell(fp);
	rv = fseek(fp, 0, SEEK_SET);

	if (rv != 0) {
		fprintf(stderr, "SEEK SET FAILED\n");
		exit(EXIT_FAILURE);
	}

	return size;
}
/*
 * Duplicates vertices to make graph undirected
 *
 */
static uint64_t *mirror(uint64_t *ed, uint64_t *ned) {

	uint64_t i, n;

	if (undirected == 1) {
	ed = (uint64_t *)Realloc(ed, (ned[0]*4)*sizeof(*ed));

	n = 0;
	for(i = 0; i < ned[0]; i++) {
		if (ed[2*i] != ed[2*i+1]) {
			ed[2*ned[0]+2*n] = ed[2*i+1];
			ed[2*ned[0]+2*n+1] = ed[2*i];
			n++;
		}
	}
	ned[0] += n;
	}
	return ed;
}

/*
 * Read graph data from file
 */
static uint64_t read_graph(int myid, int ntask, const char *fpath, uint64_t **edge) {
#define ALLOC_BLOCK     (2*1024)

	uint64_t *ed=NULL;
	uint64_t i, j;
	uint64_t n, nmax;
	uint64_t size;
	int64_t  off1, off2;

	int64_t  rem;
	FILE     *fp;
	char     str[MAX_LINE];

	fp = Fopen(fpath, "r");

	size = getFsize(fp);
	rem = size % ntask;
	off1 = (size/ntask)* myid    + (( myid    > rem)?rem: myid);
	off2 = (size/ntask)*(myid+1) + (((myid+1) > rem)?rem:(myid+1));

	if (myid < (ntask-1)) {
		fseek(fp, off2, SEEK_SET);
		fgets(str, MAX_LINE, fp);
		off2 = ftell(fp);
	}
	fseek(fp, off1, SEEK_SET);
	if (myid > 0) {
		fgets(str, MAX_LINE, fp);
		off1 = ftell(fp);
	}

	n = 0;
	nmax = ALLOC_BLOCK; // must be even
	ed = (uint64_t *)Malloc(nmax*sizeof(*ed));
	uint64_t lcounter = 0;
	uint64_t nedges = -1;
	int comment_counter = 0;

	/* read edges from file */
	while (ftell(fp) < off2) {

		// Read the whole line
		fgets(str, MAX_LINE, fp);

		// Strip # from the beginning of the line
		if (strstr(str, "#") != NULL) {
			//fprintf(stdout, "\nreading line number %"PRIu64": %s\n", lcounter, str);
			if (strstr(str, "Nodes:")) {
				sscanf(str, "# Nodes: %"PRIu64" Edges: %"PRIu64"\n", &i, &nedges);
				fprintf(stdout, "N=%"PRIu64" E=%"PRIu64"\n", i, nedges);
			}
			comment_counter++;
		} else if (str[0] != '\0') {
			lcounter ++;
			// Read edges
			sscanf(str, "%"PRIu64" %"PRIu64"\n", &i, &j);

			if (i >= N || j >= N) {
				fprintf(stderr,
						"[%d] In file %s line %"PRIu64" found invalid edge in %s for N=%"PRIu64": (%"PRIu64", %"PRIu64")\n",
						myid, fpath, lcounter, str, N, i, j);
				exit(EXIT_FAILURE);
			}

			if (n >= nmax) {
				nmax += ALLOC_BLOCK;
				ed = (uint64_t *)Realloc(ed, nmax*sizeof(*ed));
			}
			ed[n]   = i;
			ed[n+1] = j;
			n += 2;
		}
	}
	fclose(fp);

	n /= 2; // number of ints -> number of edges
	*edge = mirror(ed, &n);
	return n;
#undef ALLOC_BLOCK
}








static void dump_edges(uint64_t *ed, uint64_t nedge, const char *desc) {

	uint64_t i;
	fprintf(outdebug, "%s - %ld\n",desc, nedge);

	for (i = 0; i < nedge ; i++)
		fprintf(outdebug, "%"PRIu64"\t%"PRIu64"\n", ed[2*i], ed[2*i+1]);

	fprintf(outdebug, "\n");
	return;
}

static int cmpedge_1d(const void *p1, const void *p2) {
	  uint64_t *l1 = (uint64_t *) p1;
	  uint64_t *l2 = (uint64_t *) p2;



	  if (l1[0] < l2[0]) return -1;
	  if (l1[0] > l2[0]) return 1;

	  if (l1[1] < l2[1]) return -1;
	  if (l1[1] > l2[1]) return 1;

	  return 0;
}

/*
 * Compare Edges
 *
 * Compares Edges p1 (a,b) with p2 (c,d) according to the following algorithm:
 * First compares the nodes where edges are assigned, if they are on the same processor than compares
 * tail and than head
 */
static int cmpedge(const void *p1, const void *p2) {

	uint64_t *l1 = (uint64_t *) p1;
	uint64_t *l2 = (uint64_t *) p2;

	if (EDGE2PROC(l1[0], l1[1]) < EDGE2PROC(l2[0], l2[1]) ) return -1;
	if (EDGE2PROC(l1[0], l1[1]) > EDGE2PROC(l2[0], l2[1]) ) return 1;

	if (l1[0] < l2[0]) return -1;
	if (l1[0] > l2[0]) return 1;

	if (l1[1] < l2[1]) return -1;
	if (l1[1] > l2[1]) return 1;

	return 0;
}

static LOCINT degree_reduction(int myid, int ntask, uint64_t **ed, uint64_t nedge,
		                       uint64_t **edrem, LOCINT *ner) {
      uint64_t u=-1, v=-1, next_u= -1, next_v= -1, prev_u = -1;
      uint64_t i,j;
      uint64_t *n_ed=NULL;  // new edge list
      uint64_t *o_ed=NULL;  // removed edge list
      uint64_t *r_ed=NULL;   // input edge list

      uint64_t ncouple = 0;
      uint64_t nod=0, ne=0, pnod = 0, skipped=0; // vrem=0;

      // Sort Edges (u,v) by u
      qsort(*ed, nedge, sizeof(uint64_t[2]), cmpedge_1d);
      r_ed = *ed;

      //dump_edges(r_ed, nedge,"Degree Reduction Edges 1-D");

      n_ed = (uint64_t *)Malloc(2*nedge*sizeof(*n_ed));
      o_ed = (uint64_t *)Malloc(2*nedge*sizeof(*o_ed));

      fprintf(stdout, "[rank %d] Memory allocated\n", myid);

      for (i = 0; i < nedge-1; i++){
          u = r_ed[2*i];
          v = r_ed[2*i+1]; // current is ( u,v )    next pair is next_u, next_t based on index j
          j = 2*i + 2;
          next_u = r_ed[j];
          next_v = r_ed[j+1];

          if ((u==v) || ( (u == next_u) && (v == next_v))) { // Skip
			skipped++;
			prev_u = u;
			continue;
          }

          if ((u != next_u) && (u != prev_u)){
	    // This is a 1-degre remove
	        o_ed[2*nod] = v;
			o_ed[2*nod+1] = u;
			nod++;
          }
          else { // this is a first of a series or within a series
        	n_ed[2*ne] = u;
		n_ed[2*ne+1] = v;
		ne++;
          }
          prev_u = u;
      }
      // Check last edge
      u = r_ed[2*nedge-2];
      v = r_ed[2*nedge-1];
      if (u==prev_u) {
          n_ed[2*ne] = u;
          n_ed[2*ne+1] = v;
          ne++;
      } else if (u!=v) { // 1-degree store v before u
          o_ed[2*nod] = v;
          o_ed[2*nod+1] = u;
          nod++;
      }
  
      fprintf(stdout, "[rank %d] Edges removed during fist step %lu\n", myid, nod);

      // Sort Edges (u,v) by u
      qsort(o_ed, nod, sizeof(uint64_t[2]), cmpedge_1d);


    // remove edges for vertices removed in the previous step
    if (nod > 0){
        // Number of edges left after 1-degree removal
        nedge = ne;
        // nod is the number of edges we need to remove after exchange
        // nedge are the number of edges we
        ne=0;
		for (i = 0; i < nedge; i++) {
			// This is required to solve the case when two vertices are connected between them but
			// disconnected from all the others
			while ((n_ed[2*i] > o_ed[2*pnod]) && (pnod < nod)) {
				ncouple++;
				pnod++;
			}

			if ((n_ed[2*i] == o_ed[2*pnod]) && (n_ed[2*i+1] == o_ed[2*pnod+1])) {
				pnod++;
				// skip this
				continue;
			} else {
				// save this in the remaining edges
				r_ed[2*ne] = n_ed[2*i];
				r_ed[2*ne+1] = n_ed[2*i+1];
				ne++;
			}
		}
    } else memcpy(r_ed, n_ed, 2*ne*sizeof(r_ed));

    fprintf(stdout, "[rank %d] Edges removed during second step %lu\n", myid, pnod);
    fprintf(stdout, "[rank %d] Couple of edges removed %lu\n", myid, ncouple);

    //dump_edges(r_ed,ne, "GRAPH FOR CSC");
    *ner = pnod;    // How many vertices have been removed
    *edrem = o_ed;  // Array of removed edges
    //o_ed = NULL;  // ATTENZIONE !!
    free(n_ed);
    //free(o_ed);   //ATTENZIONE !! ???
    return ne;
}


/*
 *
 * ed   array with edges
 * ned  number of edges
 * deg  array with degrees
 *
 */

static uint64_t norm_graph(uint64_t *ed, uint64_t ned, LOCINT *deg) {

	uint64_t l, n;

	if (ned == 0) return 0;

	qsort(ed, ned, sizeof(uint64_t[2]), cmpedge);
	// record degrees considering multiple edges
	// and self-loop and remove them from edge list
	if (deg != NULL) deg[GI2LOCI(ed[0])]++;
	for(n = l = 1; n < ned; n++) {

		if (deg != NULL) deg[GI2LOCI(ed[0])]++;
		if (((ed[2*n]   != ed[2*(n-1)]  )  ||   // Check if two consecutive heads are different
			 (ed[2*n+1] != ed[2*(n-1)+1])) &&   // Check if two consecutive tails are different
			 (ed[2*n] != ed[2*n+1])) {          // It is not a "cappio"

			ed[2*l]   = ed[2*n];                // since it is not a "cappio" and is not a duplicate edge, copy it in the final edge array
			ed[2*l+1] = ed[2*n+1];
			l++;
		}
	}
	return l;
}

static void init_bc_1degree(uint64_t *edrem, uint64_t nedrem, uint64_t nverts, LOCINT * reach)
{
	uint64_t i;
	LOCINT ur = 0;
	for (i = 0; i < nedrem; i++){
	      // Edrem are edges (u,v) where v is a 1-degree vertex removed
	      ur = GI2LOCI(edrem[2*i]); // this is local row
	      // We have to use the number of vertices in the connected component
	      reach[ur]++;
	}
}

/*
 * Build compressed sparse column
 *
 */

static void build_csc(uint64_t *ed, uint64_t ned, LOCINT **col, LOCINT **row) {

	LOCINT *r, *c, *tmp, i;

	/* count edges per col */
	tmp = (LOCINT *)Malloc(col_bl*sizeof(*tmp));
	for(i = 0; i < ned; i++)
		tmp[GJ2LOCJ(ed[2*i+1])]++;  // Here we have the local degree (number of edges for each local row)

	/* compute csc col[] vector with nnz in last element */
	c = (LOCINT *)Malloc((col_bl+1)*sizeof(*c));
	c[0] = 0;
	for(i = 1; i <= col_bl; i++)
		c[i] = c[i-1] + tmp[i-1];  // Sum to the previous index the local degree.

	/* fill csc row[] vector */
	memcpy(tmp, c, col_bl*sizeof(*c)); /* no need to copy last int (nnz) */

	r = (LOCINT *)Malloc(ned*sizeof(*r));
	for(i = 0; i < ned; i++) {
		r[tmp[GJ2LOCJ(ed[2*i+1])]] = GI2LOCI(ed[2*i]);
		tmp[GJ2LOCJ(ed[2*i+1])]++;
	}
	free(tmp);

	*row = r;
	*col = c;

	return;
}


static double bc_func_mono(LOCINT *row, LOCINT *col, LOCINT *frt_all, LOCINT* frt, int* hFnum,
					       LOCINT *msk, int *lvl, LOCINT *deg,
						   LOCINT *sigma, LOCINT *frt_sig, float *delta, LOCINT rem_ed,
						   uint64_t v0, LOCINT *vRbuf, int *vRnum, LOCINT *hSbuf, int *hSnum, LOCINT *hRbuf,
						   int *hRnum, float *hSFbuf, float *hRFbuf, LOCINT *reach,
						   uint64_t* total_time, int dump) {

	int 	 level = 0, ncol;
	uint64_t nfrt=0;
	uint64_t nvisited = 1;
	double	 teps=0;


	TIMER_DEF(0);

	*total_time=0;

	memset((LOCINT *)sigma, 0, row_pp*sizeof(*sigma));
	memset((float*)delta,0,row_bl*sizeof(*delta));
	memset((int *)lvl, 0, row_pp*sizeof(*lvl));
	memset(hFnum, 0, row_bl*sizeof(*hFnum));
	memset((LOCINT *)frt, 0, row_pp*sizeof(*frt));

	LOCINT lv0 = GI2LOCI(v0);
	nfrt++;
	set_mlp_cuda(lv0, 0, 1);

	// START UPWARD BC
	TIMER_START(0);


	while(1) {
	        // We start a new BFS level
              level++;
	      hFnum[level] = hFnum[level-1] + nfrt;
	      nfrt = scan_col_csc_cuda_mono(nfrt, level);
	      if (!nfrt) break; // Exit from the loop since we do not have new vertices to visit
	      nvisited += nfrt;
	} // While(1)
	int depth = level - 2;
	ncol = 0;

	do {
	      ncol = hFnum[depth+1] - hFnum[depth]; // number of vertices to process
	      scan_frt_csc_cuda_mono(hFnum[depth], ncol, depth);
	      depth--;

	} while (depth > 0);

	// All delta values have been calculated and stored into device buffer
	LOCINT all = 0;
	if (heuristic == 1 || heuristic == 3){
	      pre_update_bc_cuda(reach, v0, &all);
	}
	all += reach[GI2LOCI(v0)]; // questo serve?
    	update_bc_cuda(v0, row_pp, nvisited+all);

	TIMER_STOP(0);
	*total_time = TIMER_ELAPSED(0);


	get_msk(msk);

	// compute teps
	
	unsigned int j,n = 0;
	for(j = 0; j < row_pp; j++)
		n += (!!MSKGET(msk,j)) * deg[j];

	n >>= 1; // ??
	teps = ((double)n)/(*total_time/1.0E+6);

//	fprintf(stdout, "\n\n\n\nElapsed time: %f secs\n", *total_time/1.0E+6);
//	fprintf(stdout, "Measured MEGA-TEPS: %lf\n", teps/(1024*1024));

	return teps/(1024*1024);
}




enum {s_minimum,
      s_firstquartile,
      s_median,
      s_thirdquartile,
      s_maximum,
      s_mean,
      s_std,
      s_LAST};

static int compare_doubles(const void* a, const void* b) {

	double aa = *((const double *)a);
	double bb = *((const double *)b);

	return (aa < bb) ? -1 : (aa == bb) ? 0 : 1;
}

static void get_statistics(const double x[], int n, double r[s_LAST]) {

	double temp;
	int i;

	/* Compute mean. */
	temp = 0;
	for(i = 0; i < n; ++i) temp += x[i];
	temp /= n;
	r[s_mean] = temp;

	/* Compute std. dev. */
	temp = 0;
	for(i = 0; i < n; ++i)
			temp += (x[i] - r[s_mean])*(x[i] - r[s_mean]);
	temp /= n-1;
	r[s_std] = sqrt(temp);
	r[s_std] /= (r[s_mean]*r[s_mean]*sqrt(n-1));

	/* Sort x. */
	double* xx = (double*)Malloc(n*sizeof(double));
	memcpy(xx, x, n*sizeof(double));
	qsort(xx, n, sizeof(double), compare_doubles);

	/* Get order statistics. */
	r[s_minimum] = xx[0];
	r[s_firstquartile] = (xx[(n-1)/4] + xx[n/4]) * .5;
	r[s_median] = (xx[(n-1)/2] + xx[n/2]) * .5;
	r[s_thirdquartile] = (xx[n-1-(n-1)/4] + xx[n-1-n/4]) * .5;
	r[s_maximum] = xx[n-1];

	/* Clean up. */
	free(xx);
}

static void print_stats(double *teps, int n) {

	int i;
	double stats[s_LAST];

	for(i = 0; i < n; i++) teps[i] = 1.0/teps[i];

	get_statistics(teps, n, stats);

	fprintf(stdout, "TEPS statistics:\n");
	fprintf(stdout, "\t   harm mean: %lf\n", 1.0/stats[s_mean]);
	fprintf(stdout, "\t   harm stdev: %lf\n", stats[s_std]);
	fprintf(stdout, "\t   median: %lf\n", 1.0/stats[s_median]);
	fprintf(stdout, "\t   minimum: %lf\n", 1.0/stats[s_maximum]);
	fprintf(stdout, "\t   maximum: %lf\n", 1.0/stats[s_minimum]);
	fprintf(stdout, "\tfirstquartile: %lf\n", 1.0/stats[s_firstquartile]);
	fprintf(stdout, "\tthirdquartile: %lf\n", 1.0/stats[s_thirdquartile]);
	return;
}

void usage(const char *pname) {

	prexit("Usage (BC-1GPU):\n"
			"\t %1$s -p 1x1 [-o outfile] [-D] [-d] [-m] [-N <# of searches>]\n"
		    "\t -> to visit a graph read from file:\n"
		   "\t\t -f <graph file> -n <# vertices> [-r <start vert>]\n"
		   "\t -> to visit an RMAT graph:\n"
		   "\t\t -S <scale> [-E <edge factor>] DISABLED\n"
			"\t Where:\n"
			"\t\t -D to ENABLE debug information\n"
			"\t\t -m to DISABLE mono GPU optimization\n"
			"\t\t -U DO NOT make graph Undirected\n"
			"\n", pname);
	return;
}

int main(int argc, char *argv[]) {

	int gread=-1;
	int scale=21, edgef=16; // 1 1-degree reduction
	short  debug = 0;
	int64_t  i, j;
	uint64_t nbfs=1, ui;
	LOCINT n, l, ned, rem_ed = 0;

	uint64_t *edge=NULL;
	uint64_t *rem_edge=NULL;

	LOCINT  *col=NULL;
	LOCINT  *row=NULL;
	LOCINT  *frt=NULL;
	LOCINT  *frt_all=NULL;
	int *hFnum = NULL; /* offsets of ventices in the frontier for each level */


	LOCINT *degree=NULL; // Degree for all vertices in the same column
	LOCINT *sigma=NULL;  // Sigma (number of SP)
	LOCINT *frt_sigma=NULL;  // Sigma (number of SP)

	float   *hRFbuf=NULL;
	float   *hSFbuf=NULL;
	float   *delta=NULL;
	float   *bc_val=NULL;
	LOCINT  *msk=NULL;
	int     *lvl=NULL; //, level;
	LOCINT  *deg=NULL;

	LOCINT  *vRbuf=NULL;
	int     *vRnum=NULL; /* type int (MPI_Send()/Recv() assumes int counts) */

	LOCINT  *hSbuf=NULL;
	LOCINT  *hRbuf=NULL;
	int     *hSnum=NULL;
	int     *hRnum=NULL;
	LOCINT  *prd=NULL; /* predecessors array */

	uint64_t v0 = 0, startv = 0;

	int rootset = 0;

	int cntask;
	char *gfile=NULL, *p=NULL, *ofile=NULL;
	signed char opt;

	TIMER_DEF(0);

	double *teps=NULL;

	int random = 0;

	//      struct timespec mytime;
//      clock_gettime( CLOCK_REALTIME, &mytime);


	if (argc == 1) {
		usage(argv[0]);
		exit(EXIT_FAILURE);
	}
	outId = time(NULL);

	int wr, pl=0, limit = sizeof(cmdLine);
	wr = snprintf (cmdLine+pl, limit, "MPI Tasks %d\n", gntask);
	if (wr<0) exit(EXIT_FAILURE);
	limit -= wr; pl +=wr;
	for (i = 0; i < argc; i++) {
		wr = snprintf (cmdLine+pl, limit," %s", argv[i]);
		if (wr<0) exit(EXIT_FAILURE);
		limit -= wr; pl +=wr;
	}
	snprintf (cmdLine+pl, limit,"\n");

	if (gmyid==0) {
		fprintf(stdout,"%s\n",cmdLine);
	}

	while((opt = getopt(argc, argv, "o:p:amhDR:Uf:n:r:S:E:N:H:")) != EOF) {
#define CHECKRTYPE(exitval,opt) {\
		if (exitval == gread) prexit("Unexpected option -%c!\n", opt);\
				else gread = !exitval;\
		}
		switch (opt) {
			case 'H' :
					if (0 == sscanf(optarg, "%d", &heuristic)) prexit("Invalid Heuristic Option (-H): %s\n", optarg);					
					if ( heuristic >= 2 && ntask > 1) prexit("2-degree Heuristic is allowed in single-gpu configuration (-H): %s\n", optarg);
					break;
					//heuristic selection
			case 'o' :
					ofile = strdup(optarg);
					break;
			case 'p':
					strncpy(strmesh,optarg,10);
					p = strtok(optarg, "x");
					if (!p) prexit("Invalid proc mesh field.\n");
					if (0 == sscanf(p, "%d", &R)) prexit("Invalid number of rows for proc mesh (-p): %s\n", p);
					p = strtok(NULL, "x");
					if (!p) prexit("Invalid proc mesh field.\n");
					if (0 == sscanf(p, "%d", &C)) prexit("Invalid number of columns for proc mesh (-p): %s\n", p);
					break;
    			case 'R':
					// Random Root
					sscanf(optarg, "%d", &random);
					if (random != 0 && random != 1)
						prexit("Invalid random option (-R): %s\n", optarg);
					break;
			case 'D':
					// DEBUG
					debug = 1;
					break;
			case 'm':
					// Mono-Multi GPU
					mono = 0;
					break;
			case 'f':
					CHECKRTYPE(0, 'f')
					gfile = strdup(optarg);
					break;
			case 'n':
					CHECKRTYPE(0, 'n')
					if (0 == sscanf(optarg, "%"PRIu64, &N)) prexit("Invalid number of vertices (-n): %s\n", optarg);
					break;
			case 'r':
					//CHECKRTYPE(0, 'r')
					if (0 == sscanf(optarg, "%"PRIu64, &startv)) prexit("Invalid root vertex (-r): %s\n", optarg);
					rootset = 1;
					break;
			case 'S':
					CHECKRTYPE(1, 'S')
					if (0 == sscanf(optarg, "%d", &scale)) prexit("Invalid scale (-S): %s\n", optarg);
					N = ((uint64_t) 1) << scale;
					break;
			case 'E':
					CHECKRTYPE(1, 'E')
					if (0 == sscanf(optarg, "%d", &edgef)) prexit("Invalid edge factor (-S): %s\n", optarg);
					break;
			case 'N':
					if (0 == sscanf(optarg, "%ld", &nbfs)) prexit("Invalid number of bfs (-N): %s\n", optarg);
					break;
			case 'U':
					// Undirected
					undirected = 0;
					break;
			case 'h':
			case '?':
			default:
					usage(argv[0]);
					exit(EXIT_FAILURE);
		}
#undef CHECKRTYPE
	}

	if (gread) {
		if (!gfile || !N)
			prexit("Graph file (-f) and number of vertices (-n)"
				   " must be specified for file based bfs.\n");
	}

	if (0 >= R || MAX_PROC_I < R || 0 >= C || MAX_PROC_J < C)
		prexit("R and C must be in range [1,%d] and [1,%d], respectively.\n", MAX_PROC_I, MAX_PROC_J);

	if (0 != N%(R*C))
		prexit("N must be multiple of both R and C.\n");

	ntask=1; // FORCED 1GPU
	cntask = 1;

#ifndef _LARGE_LVERTS_NUM
	if ((N/R) > UINT32_MAX) {
		prexit("Number of vertics per processor too big (%"LOCPRI"), please"
			   "define _LARGE_LVERTS_NUM macro in %s.\n", (N/R), __FILE__);
	}
#endif

	if (startv >= N)
		prexit("Invalid start vertex: %"PRIu64".\n", startv);


	row_bl = N/(R*C); /* adjacency matrix rows per block:    N/(RC) */
	col_bl = N/C;     /* adjacency matrix columns per block: N/C */
	row_pp = N/R;     /* adjacency matrix rows per proc:     N/(RC)*C = N/R */

	if ((gmyid==0) && (debug==1)) {
	  char     fname[MAX_LINE];
	  snprintf(fname, MAX_LINE, "%s_%d.log", "debug", gmyid);
	  outdebug = Fopen(fname,"w");
	}

	char *resname = NULL;

	// Disable random when a starting node is provided. Random function is removed.
	if (rootset > 0) random = 0;

	if (gmyid == 0) {
		fprintf(stdout,"\n\n****** DEVEL VERSION ******\n\n\n\n***************************\n\n");
		fprintf(stdout, "Total number of vertices (N): %"PRIu64"\n", N);
		fprintf(stdout, "Processor mesh rows (R): %d\n", R);
		fprintf(stdout, "Processor mesh columns (C): %d\n", C);
		if (gread) {
			fprintf(stdout, "Reading graph from file: %s\n", gfile);
		} else {
			fprintf(stdout, "RMAT graph scale: %d\n", scale);
			fprintf(stdout, "RMAT graph edge factor: %d\n", edgef);
			fprintf(stdout, "Number of bc rounds: %ld\n", nbfs);
			if (random) fprintf(stdout, "Random mode\n");
			else fprintf(stdout, "First node: %lu\n", startv);
		}
                fprintf(stdout,"\n\n");
                if (heuristic == 0){
                      fprintf(stdout, "HEURISTICs: OFF: %d\n", heuristic);

                }
                else if (heuristic == 1){
			fprintf(stdout, "HEURISTICs: 1-degree reduction ON: %d\n", heuristic);
		}
		
                fprintf(stdout,"PREFIX SCAN library: THRUST\n");

#ifdef OVERLAP
		fprintf(stdout,"OVERLAP: ON\n");
#else
		fprintf(stdout,"OVERLAP: OFF\n");
#endif
#ifdef ONEPREFIX
      	        fprintf(stdout,"PREFIX SCAN optimization: ON\n");
#else
      	        fprintf(stdout,"PREFIX SCAN optimization: OFF\n");
#endif

                fprintf(stdout,"\n");
	}

	if (NULL != ofile) {
		fprintf(stdout, "Result written to file: %s\n", ofile);
		resname = (char*) malloc((sizeof(ofile)+MAX_LINE)*sizeof(*resname));
		sprintf(resname, "%s_%dX%d_%d.log",ofile, R,C, gmyid);
	}

	/* fill processor mesh */
	memset(pmesh, -1, sizeof(pmesh));
	for(i = 0; i < R; i++)
		for(j = 0; j < C; j++)
			pmesh[i][j] = i*C + j;

	TIMER_START(0);
	ned = read_graph(myid, ntask, gfile, &edge); // Read from file
	TIMER_STOP(0);
	if (myid == 0) fprintf(stdout, " done in %f secs\n", TIMER_ELAPSED(0)/1.0E+6);

	if (heuristic != 0){
		l = norm_graph(edge, ned, NULL);
		ned = l;
	}


	// 1 DEGREE PREPROCESSING TIMING ON
	if (heuristic == 1 || heuristic == 3){
		if (gmyid == 0) fprintf(stdout, "Degree reduction graph (%d)...\n", heuristic);
		TIMER_START(0);
		// DEGREE REDUCTION - Edge Based
		ned = degree_reduction(myid, ntask, &edge, ned, &rem_edge, &rem_ed);
		TIMER_STOP(0);
		degree_reduction_time = TIMER_ELAPSED(0);
	}
#ifndef _LARGE_LVERTS_NUM
	if (ned > UINT32_MAX) {
		fprintf(stderr,"Too many vertices assigned to me. Change LOCINT\n");
		exit(EXIT_FAILURE);
	}
#endif
	//if (myid == 0) fprintf(stdout, "task %d Removing multi-edges...",gmyid);
	deg = (LOCINT *)Malloc(row_pp*sizeof(*deg));

	TIMER_START(0);
	// Normalize graph: remove loops and duplicates edges cappi o loops?
	// THIS ALSO CALCULATES DEGREES
	l = norm_graph(edge, ned, deg);
	TIMER_STOP(0);
	if (myid == 0) fprintf(stdout, "task %d done in %f secs\n", gmyid, TIMER_ELAPSED(0)/1.0E+6);
	ned = l;

	if (myid == 0) fprintf(stdout, "task %d, Creating CSC...", gmyid);
	TIMER_START(0);

	build_csc(edge, ned, &col, &row);  // Make the CSC Structure
	TIMER_STOP(0);
	if (myid == 0) fprintf(stdout, "task %d done in %f secs\n", gmyid, TIMER_ELAPSED(0)/1.0E+6);
	freeMem(edge);
	freeMem(deg);
	n = initcuda(ned, col, row);

	// Allocate The frontier
	frt = (LOCINT *)CudaMallocHostSet(row_bl*sizeof(*frt),0);
	frt_all = (LOCINT *)CudaMallocHostSet(MAX(col_bl, row_pp)*sizeof(*frt), 0);
	// Allocate The BFS level
	lvl = (int *)CudaMallocHostSet(row_pp*sizeof(*lvl),0);
	//	cudaHostRegister(lvl, row_pp*sizeof(*lvl), 0);
	// Allocate the frontier offset for each level
	hFnum = (int*)CudaMallocHostSet(MAX(row_pp, col_bl)*sizeof(*hFnum),0);
	// Allocate Degree array
	degree = (LOCINT *)CudaMallocHostSet(col_bl*sizeof(*degree),0);
	// Allocate sigma
	sigma = (LOCINT *)CudaMallocHostSet(row_pp*sizeof(*sigma),0);
	//	cudaHostRegister(sigma, row_pp*sizeof(*sigma), 0);
	// Allocate Frontier Sigma
	frt_sigma = (LOCINT *)CudaMallocHostSet(row_bl*sizeof(*frt_sigma),0);
	// Allocate delta
	delta = (float*)CudaMallocHostSet(row_bl*sizeof(*delta),0);
	// Allocate BC
	bc_val = (float*)CudaMallocHostSet(row_pp*sizeof(*bc_val),0);
	reach = (LOCINT*)CudaMallocHostSet(row_pp*sizeof(*reach),0);
	if (heuristic == 1){
		//if(myid==0) printf("task %d edges removed %d ...\n",gmyid,rem_ed);
		init_bc_1degree(rem_edge, rem_ed, N, reach);
		if(myid==0) printf("task %d Total edges removed %d\n",gmyid, rem_ed);
	}
	
	
    	get_deg(degree);
	init_bc_1degree_device(reach);

	// Allocate BitMask to store visited unique vertices ???
	msk = (LOCINT *)Malloc(((row_pp+BITS(msk)-1)/BITS(msk))*sizeof(*msk));
	nbfs = MIN(N, nbfs);

#ifdef _FINE_TIMINGS
    // Allocate for statistical data
    mystats = (STATDATA*)Malloc(N*sizeof(STATDATA));
    memset(mystats, 0, N*sizeof(STATDATA));
#endif

#ifdef ONEPREFIX
	tlvl = (LOCINT*)Malloc(MAX_LVL*sizeof(*tlvl));
#endif

	LOCINT skip, reach_v0, nrounds=0, skipped=0;
	uint64_t all_time=0, bc_time=0, min_time=UINT_MAX, max_time=0;
        double mteps = 0;
	uint64_t commu_all_time=0, commu_time=0, compu_all_time=0, compu_time=0;
	if (myid == 0) fprintf(stdout, "task %d: BC computation is running...\n", gmyid);
        if (heuristic > 1) fprintf(stdout,"REMOVED\n");
	else{// NO H 2 o H 3
		for(ui = 0; ui < nbfs; ui += cntask) {
			v0 = startv + ui;
			skip = 0;
			bc_time = 0;
			reach_v0 = 0;
			if (VERT2PROC(v0) == myid) {
			//fprintf(stdout,"Root = %lu; TaskId=%d; LocalId=%d\n", v0, myid, GJ2LOCJ(v0));

				// Check v0 degree
				if (degree[GJ2LOCJ(v0)]==0) {
					skip=1;
				}
				reach_v0 = reach[GI2LOCI(v0)];
			}
			if (skip) {
				//teps[ui] = 0;
					skipped++;
					continue;
			}

			setcuda(ned, col, row, reach_v0);

			mteps += bc_func_mono(row, col,  frt_all, frt,  hFnum, msk,   lvl, degree,  sigma, frt_sigma, delta, rem_ed,
							 v0, vRbuf,  vRnum, hSbuf, hSnum, hRbuf, hRnum, hSFbuf, hRFbuf, reach,
								  &bc_time, 0);
                        nrounds++;
			all_time += bc_time;
			commu_all_time += commu_time;
			compu_all_time += compu_time;

			if (bc_time > max_time ) max_time = bc_time;
			if (bc_time < min_time && bc_time != 0 ) min_time= bc_time;
		}
	}

        TIMER_START(0);
	get_bc(bc_val);
        TIMER_STOP(0);
	uint64_t bcred_time = TIMER_ELAPSED(0);

	if (mycol == 0 && resname != NULL) {
		FILE *resout = fopen(resname,"w");

		fprintf(resout,"BC RESULTS\n");
		fprintf(resout,"ROWPP %u\n", row_pp);
		fprintf(resout,"NodeId\tBC_VAL\n");
		LOCINT k;
		for (k=0;k<row_pp;k++) {
			fprintf(resout,"%d\t%.2f\n", LOCI2GI(k) ,bc_val[k]/2.0);
		}
		fclose(resout);
	}

	if (myid == 0) {

		fprintf(stdout, "\n------------- RESULTS ---------------\n");

		fprintf(stdout,"ClusterId\tSkip\tRounds\tExecTime\tRoundsTime\tMax\tMin\tMean\t1-dReduTime\n");
		fprintf(stdout,"%d\t%d\t%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n",
				 	 	 	 	 	 	 	 0, skipped, nrounds,
				                                                        (all_time+bcred_time)/1.0E+6,
											 all_time/1.0E+6,
											 max_time/1.0E+6,
											 min_time/1.0E+6,
											 all_time/1.0E+6/nrounds,
											 degree_reduction_time/1.0E+6);

		fprintf(stdout,"task %d BC skipped: %d \n", gmyid, skipped);
		fprintf(stdout,"task %d BC rounds: %d  \n", gmyid, nrounds);
		fprintf(stdout,"task %d BC execution total time: %lf sec\n",gmyid,(all_time+bcred_time)/1.0E+6);
		fprintf(stdout,"task %d BC rounds total time: %lf sec\n",gmyid,all_time/1.0E+6);
		fprintf(stdout,"task %d BC Max time: %lf sec\n",gmyid, max_time/1.0E+6);
		fprintf(stdout,"task %d BC Min time: %lf sec\n",gmyid, min_time/1.0E+6);
		fprintf(stdout,"task %d BC Mean time: %lf sec\n",gmyid, all_time/1.0E+6/nrounds);
                fprintf(stdout,"task %d BC M-TEPS: %lf sec\n",gmyid, mteps/(double)nrounds);
		if (nbfs < N){
			unsigned int vskipped = 0;
			vskipped=(N-nbfs)*skipped/nbfs;
			double avg = all_time/1.0E+6/nrounds; 
			fprintf(stdout,"task %d BC simulated time: %lf sec (virtual skipped %d)\n",gmyid, avg*(N-skipped-vskipped)/cntask, vskipped);

		}
		if ( heuristic == 1 || heuristic == 3 ){
			fprintf(stdout,"task %d 1-Degree reduction: %lf sec\n",gmyid, degree_reduction_time/1.0E+6);
		}
		fprintf(stdout, "\n");
	}

	
	if (outdebug!=NULL) fclose(outdebug);

	freeMem(col);
	freeMem(row);
	freeMem(mystats);
	freeMem(gfile);
	freeMem(teps);
	fincuda();
	CudaFreeHost(lvl);
	CudaFreeHost(sigma);
	CudaFreeHost(frt);
	CudaFreeHost(frt_all);
	CudaFreeHost(degree);
	CudaFreeHost(frt_sigma);
	CudaFreeHost(hFnum);
	CudaFreeHost(delta);
	CudaFreeHost(bc_val);
	CudaFreeHost(reach);
	CudaFreeHost(prd);
	CudaFreeHost(vRbuf);
	CudaFreeHost(vRnum);
	CudaFreeHost(hSbuf);
	CudaFreeHost(hSnum);
	CudaFreeHost(hRbuf);
	CudaFreeHost(hRnum);
	CudaFreeHost(hSFbuf);
	CudaFreeHost(hRFbuf);

//ONEPREFIX
	freeMem(tlvl);
	freeMem(tlvl_v1);

        return 0;
}

