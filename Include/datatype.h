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
    int r;
};
struct shortcut_Config {
    int n_chnls;
    int H, W;
    int r;
};
struct relu_Config {
    int n_chnls;
    int H, W;
    int r;
};
struct relupooling_Config {
    int n_chnls;
    int sz_kern;
    int stride;
    int padding;
    int Hi, Wi, Ho, Wo;
    int mode;
    int r;
};

union lyrConfig {
    struct conv_Config* conv_conf;
    struct shortcut_Config* shortcut_conf;
    struct relu_Config* relu_conf;
    struct relupooling_Config* relupooling_conf;
};

#endif
