//
// Created by hsiaochuan on 2026/09/04.
//
// RTKLIB computes the pseudorange it actually feeds to rescode() in prange(), which removes
// the differential code bias and the broadcast group delay of the tracked signal.  prange()
// and its helper gettgd() are `static` inside pntpos.c, so there is no way to call them; the
// two functions are copied here verbatim (only `static` -> `inline` and the trace() calls
// dropped) so that the factor graph screens and corrects its observations with exactly the
// same code RTKLIB's own point positioning uses.
//
// Source: RTKLIB third_party/RTKLIB/src/pntpos.c, Copyright (C) 2007-2020 by T.TAKASU.
//

#ifndef FASTER_LIO_RTKLIB_PRANGE_H
#define FASTER_LIO_RTKLIB_PRANGE_H

#include "rtklib_.h"

namespace rtklib {

#define SQR(x) ((x) * (x))

#define ERR_CBIAS 0.3 /* code bias error Std (m) */

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

#undef SQR
#undef ERR_CBIAS

}  // namespace rtklib_pntpos

#endif  // FASTER_LIO_RTKLIB_PRANGE_H
