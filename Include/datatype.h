#ifndef _DATATYPE_H_
#define _DATATYPE_H_

typedef int BOOL;

struct conv_Config {
    int n_ichnls;
    int n_ochnls;
    int sz_kern;
    int stride;
    int padding;
    int Hi, Wi, Ho, Wo;
};
struct relu_Config {
    int n_chnls;
};
struct relupooling_Config {
    int n_chnls;
    int sz_kern;
    int stride;
    int padding;
    int Hi, Wi, Ho, Wo;
    int mode;
};

union lyrConfig {
    struct conv_Config* conv_conf;
    struct relu_Config* relu_conf;
    struct relupooling_Config* relupooling_conf;
};

#endif
