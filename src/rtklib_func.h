//
// Created by hsiaochuan on 2026/09/04.
//

#ifndef FASTER_LIO_RTKLIB_PRANGE_H
#define FASTER_LIO_RTKLIB_PRANGE_H

#include "rtklib_.h"

namespace rtklib {

#define SQR(x) ((x) * (x))

#define ERR_CBIAS 0.3 /* code bias error Std (m) */
#define MAX(x,y)    ((x)>=(y)?(x):(y))
#define MIN_EL      (5.0*D2R)   /* min elevation for measurement error (rad) */

/* get group delay parameter (m) ---------------------------------------------*/
inline double gettgd(int sat, const nav_t *nav, int type)
{
    int i,sys=satsys(sat,NULL);

    if (sys==SYS_GLO) {
        for (i=0;i<nav->ng;i++) {
            if (nav->geph[i].sat==sat) break;
        }
        return (i>=nav->ng)?0.0:-nav->geph[i].dtaun*CLIGHT;
    }
    else {
        for (i=0;i<nav->n;i++) {
            if (nav->eph[i].sat==sat) break;
        }
        return (i>=nav->n)?0.0:nav->eph[i].tgd[type]*CLIGHT;
    }
}
/* iono-free or "pseudo iono-free" pseudorange with code bias correction -----*/
inline double prange(const obsd_t *obs, const nav_t *nav, const prcopt_t *opt,
                     double *var)
{
    double P1,P2,gamma,b1,b2;
    int sat,sys,f2;

    sat=obs->sat;
    sys=satsys(sat,NULL);
    P1=obs->P[0];
    f2=seliflc(opt->nf,satsys(obs->sat,NULL));
    P2=obs->P[f2];
    *var=0.0;

    if (P1==0.0||(opt->ionoopt==IONOOPT_IFLC&&P2==0.0)) return 0.0;
    P1-=code2bias(nav,sys,sat,obs->code[0],0);  /* L1 differential code bias */
    if (sys!=SYS_GAL||f2!=1) { /* skip GAL P2 code bias for now to match previous version */
        P2-=code2bias(nav,sys,sat,obs->code[f2],0); /* L2 or L5 differential code bias */
    }

    if (opt->ionoopt==IONOOPT_IFLC) { /* dual-frequency */

        if (sys==SYS_GPS||sys==SYS_QZS) { /* L1-L2 or L1-L5 */
            gamma=f2==1?SQR(FREQL1/FREQL2):SQR(FREQL1/FREQL5);
            return (P2-gamma*P1)/(1.0-gamma);
        }
        else if (sys==SYS_GLO) { /* G1-G2 or G1-G3 */
            gamma=f2==1?SQR(FREQ1_GLO/FREQ2_GLO):SQR(FREQ1_GLO/FREQ3_GLO);
            return (P2-gamma*P1)/(1.0-gamma);
        }
        else if (sys==SYS_GAL) { /* E1-E5b, E1-E5a */
            gamma=f2==1?SQR(FREQL1/FREQE5b):SQR(FREQL1/FREQL5);
            if (f2==1&&getseleph(SYS_GAL)) { /* F/NAV */
                P2-=gettgd(sat,nav,0)-gettgd(sat,nav,1); /* BGD_E5aE5b */
            }
            return (P2-gamma*P1)/(1.0-gamma);
        }
        else if (sys==SYS_CMP) { /* B1-B2 */
            gamma=SQR(((obs->code[0]==CODE_L2I)?FREQ1_CMP:FREQL1)/FREQ2_CMP);
            if      (obs->code[0]==CODE_L2I) b1=gettgd(sat,nav,0); /* TGD_B1I */
            else if (obs->code[0]==CODE_L1P) b1=gettgd(sat,nav,2); /* TGD_B1Cp */
            else b1=gettgd(sat,nav,2)+gettgd(sat,nav,4); /* TGD_B1Cp+ISC_B1Cd */
            b2=gettgd(sat,nav,1); /* TGD_B2I/B2bI (m) */
            return ((P2-gamma*P1)-(b2-gamma*b1))/(1.0-gamma);
        }
        else if (sys==SYS_IRN) { /* L5-S */
            gamma=SQR(FREQL5/FREQs);
            return (P2-gamma*P1)/(1.0-gamma);
        }
    }
    else { /* single-freq (L1/E1/B1) */
        *var=SQR(ERR_CBIAS);

        if (sys==SYS_GPS||sys==SYS_QZS) { /* L1 */
            b1=gettgd(sat,nav,0); /* TGD (m) */
            return P1-b1;
        }
        else if (sys==SYS_GLO) { /* G1 */
            gamma=SQR(FREQ1_GLO/FREQ2_GLO);
            b1=gettgd(sat,nav,0); /* -dtaun (m) */
            return P1-b1/(gamma-1.0);
        }
        else if (sys==SYS_GAL) { /* E1 */
            if (getseleph(SYS_GAL)) b1=gettgd(sat,nav,0); /* BGD_E1E5a */
            else                    b1=gettgd(sat,nav,1); /* BGD_E1E5b */
            return P1-b1;
        }
        else if (sys==SYS_CMP) { /* B1I/B1Cp/B1Cd */
            if      (obs->code[0]==CODE_L2I) b1=gettgd(sat,nav,0); /* TGD_B1I */
            else if (obs->code[0]==CODE_L1P) b1=gettgd(sat,nav,2); /* TGD_B1Cp */
            else b1=gettgd(sat,nav,2)+gettgd(sat,nav,4); /* TGD_B1Cp+ISC_B1Cd */
            return P1-b1;
        }
        else if (sys==SYS_IRN) { /* L5 */
            gamma=SQR(FREQs/FREQL5);
            b1=gettgd(sat,nav,0); /* TGD (m) */
            return P1-gamma*b1;
        }
    }
    return P1;
}

/* pseudorange measurement error variance ------------------------------------*/
inline double varerr(const prcopt_t *opt, const obsd_t *obs, double el, int sys)
{
    double fact=1.0,varr;

    switch (sys) {
        case SYS_GPS: fact *= EFACT_GPS; break;
        case SYS_GLO: fact *= EFACT_GLO; break;
        case SYS_SBS: fact *= EFACT_SBS; break;
        case SYS_CMP: fact *= EFACT_CMP; break;
        case SYS_QZS: fact *= EFACT_QZS; break;
        case SYS_IRN: fact *= EFACT_IRN; break;
        default:      fact *= EFACT_GPS; break;
    }
    if (el<MIN_EL) el=MIN_EL;
    /* var = R^2*(a^2 + (b^2/sin(el) + c^2*(10^(0.1*(snr_max-snr)))) + (d*rcv_std)^2) */
    varr=SQR(opt->err[1])+SQR(opt->err[2])/sin(el);
    if (opt->err[6]>0.0) {  /* if snr term not zero */
        varr+=SQR(opt->err[6])*pow(10,0.1*MAX(opt->err[5]-obs->SNR[0],0));
    }
    varr*=SQR(opt->eratio[0]);
    if (opt->err[7]>0.0) {
        /* NOTE: err[7] defaults to 0 (disabled); if it's ever enabled, this Pstd term would
         * double-count with gnss_processor.cpp's own Hatch-filtered code_var, which also uses
         * obs->Pstd. Revisit code_var there before enabling err[7]. */
        varr+=SQR(opt->err[7]*obs->Pstd[0]);
    }
    if (opt->ionoopt==IONOOPT_IFLC) varr*=SQR(3.0); /* iono-free */
    return SQR(fact)*varr;
}

#undef SQR
#undef ERR_CBIAS
#undef MAX
#undef MIN_EL

}  // namespace rtklib_pntpos

#endif  // FASTER_LIO_RTKLIB_PRANGE_H
