// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
/// \file lambdaJetPolarizationIonsDerived.cxx
/// \brief Lambda and antiLambda polarization analysis task using derived data
/// \author Cicero Domenico Muncinelli <cicero.domenico.muncinelli@cern.ch>, Campinas State University
//
// Jet Polarization Ions task -- Derived data
// ================
//
// This code loops over custom derived data tables defined on
// lambdaJetPolarizationIons.h (JetsRing, LambdaLikeV0sRing).
// From this derived data, calculates polarization on an EbE
// basis (see TProfiles).
// Signal extraction is done out of the framework, based on
// the AnalysisResults of this code.
//
//
//    Comments, questions, complaints, suggestions?
//    Please write to:
//    cicero.domenico.muncinelli@cern.ch
//

#include "PWGLF/DataModel/lambdaJetPolarizationIons.h"

#include "Common/Core/RecoDecay.h"

#include <CommonConstants/MathConstants.h>
#include <CommonConstants/PhysicsConstants.h>
#include <Framework/ASoA.h>
#include <Framework/AnalysisDataModel.h>
#include <Framework/AnalysisTask.h>
#include <Framework/BinningPolicy.h>
#include <Framework/Configurable.h>
#include <Framework/GroupedCombinations.h>
#include <Framework/HistogramRegistry.h>
#include <Framework/HistogramSpec.h>
#include <Framework/InitContext.h>
#include <Framework/Logger.h>
#include <Framework/OutputObjHeader.h>
#include <Framework/runDataProcessing.h>

#include <Math/GenVector/VectorUtil.h>
#include <Math/Vector3Dfwd.h>
#include <Math/Vector4D.h> // IWYU pragma: keep (do not replace with Math/Vector4Dfwd.h)
#include <Math/Vector4Dfwd.h>
#include <TAxis.h>
#include <TH1.h>
#include <TH2.h>
#include <TProfile.h>
#include <TProfile2D.h>
#include <TRandom3.h> // For perpendicular jet direction QAs

#include <algorithm> // std::fill, for resetting the Delta Method accumulators
#include <array>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using ROOT::Math::PtEtaPhiMVector;
using ROOT::Math::XYZVector;

// Declaring constants:
constexpr double ProtonMass = o2::constants::physics::MassProton; // Assumes particle identification for daughter is perfect
constexpr double LambdaMass = o2::constants::physics::MassLambda;
constexpr double LambdaWeakDecayConstant = 0.749;                 // DPG 2025 update
constexpr double AntiLambdaWeakDecayConstant = -0.758;            // DPG 2025 update
constexpr double PolPrefactorLambda = 3.0 / LambdaWeakDecayConstant;
constexpr double PolPrefactorAntiLambda = 3.0 / AntiLambdaWeakDecayConstant;
// Signal-extraction estimates for InMassPeak Vs OutOfMassPeak's cheap signal vs bacgkround study:
constexpr double LambdaMassSigma = 0.0017127606;

enum CentEstimator {
  kCentFT0C = 0,
  kCentFT0M,
  kCentFV0A
};

// Helper macro to avoid writing the histogram fills 4 times for about 20 histograms:
#define RING_OBSERVABLE_FILL_LIST(X, FOLDER)                                                                               \
  /* Counters */                                                                                                           \
  X(FOLDER "/LeadJet/QA/hDeltaPhi", ringLeadJet.deltaPhi)                                                                                   \
  X(FOLDER "/LeadJet/QA/h2dDeltaPhiVsDeltaEta", ringLeadJet.deltaPhi, deltaEtaLeadJet)                                                            \
  X(FOLDER "/LeadJet/QA/hDeltaTheta", ringLeadJet.deltaTheta)                                                                               \
  X(FOLDER "/LeadJet/QA/hCosDeltaTheta", ringLeadJet.cosDeltaTheta)                                                                         \
  X(FOLDER "/LeadJet/QA/hIntegrated", 0.)                                                                                          \
  X(FOLDER "/LeadJet/QA/hProxyPt", leadJet.pt)                                                                                     \
  /* Lambda pT variation -- Youpeng's proposal */                                                                          \
  X(FOLDER "/LeadJet/QA/hLambdaPt", v0pt)                                                                                          \
  /* Counters */                                                                                                           \
  X(FOLDER "/LeadJet/QA/h2dDeltaPhiVsLambdaPt", ringLeadJet.deltaPhi, v0pt)                                                                 \
  X(FOLDER "/LeadJet/QA/h2dDeltaThetaVsLambdaPt", ringLeadJet.deltaTheta, v0pt)                                                             \
  X(FOLDER "/LeadJet/QA/h2dDeltaPhiVsProxyPhi", ringLeadJet.deltaPhi, leadJet.phi)                                                        \
  /* Additional plots for instant gratification - 1D Profiles */                                                           \
  X(FOLDER "/LeadJet/hRingObservableCounts", ringLeadJet.ringObservable)                                                                       \
  X(FOLDER "/LeadJet/pRingObservableDeltaPhi", ringLeadJet.deltaPhi, ringLeadJet.ringObservable)                                                        \
  X(FOLDER "/LeadJet/pRingObservableProxyPhi", leadJet.phi, ringLeadJet.ringObservable)                                                        \
  X(FOLDER "/LeadJet/pRingObservablePhiLambda", v0phi, ringLeadJet.ringObservable)                                                             \
  X(FOLDER "/LeadJet/pRingObservableDeltaTheta", ringLeadJet.deltaTheta, ringLeadJet.ringObservable)                                                    \
  X(FOLDER "/LeadJet/EtaDependence/pRingObservableEtaLambda", v0eta, ringLeadJet.ringObservable)                                               \
  X(FOLDER "/LeadJet/EtaDependence/pRingObservableProxyEta", leadJet.eta, ringLeadJet.ringObservable)                                          \
  X(FOLDER "/LeadJet/EtaDependence/pRingObservableProxyEtaHighEtaRes", leadJet.eta, ringLeadJet.ringObservable)                                \
  X(FOLDER "/LeadJet/pRingObservableIntegrated", 0., ringLeadJet.ringObservable)                                                               \
  X(FOLDER "/LeadJet/pRingObservableLambdaPt", v0pt, ringLeadJet.ringObservable)                                                               \
  X(FOLDER "/LeadJet/pRingObservablePVz", collisionPVz, ringLeadJet.ringObservable)                                                     \
  X(FOLDER "/LeadJet/ProxyPtDependence/pRingVsProxyPt", leadJet.pt, ringLeadJet.ringObservable)                                                \
  X(FOLDER "/LeadJet/ProxyPtDependence/p2dRingVsProxyPtVsProxyEta", leadJet.pt, leadJet.eta, ringLeadJet.ringObservable)                         \
  X(FOLDER "/LeadJet/ProxyPtDependence/p2dRingVsProxyPtVsEtaV0", leadJet.pt, v0eta, ringLeadJet.ringObservable)                                  \
  X(FOLDER "/LeadJet/ProxyPtDependence/p2dRingVsProxyPtVsCentrality", leadJet.pt, centrality, ringLeadJet.ringObservable)                        \
  /* 2D Profiles */                                                                                                        \
  X(FOLDER "/LeadJet/p2dRingObservableDeltaPhiVsLambdaPt", ringLeadJet.deltaPhi, v0pt, ringLeadJet.ringObservable)                                      \
  X(FOLDER "/LeadJet/p2dRingObservableDeltaThetaVsLambdaPt", ringLeadJet.deltaTheta, v0pt, ringLeadJet.ringObservable)                                  \
  X(FOLDER "/LeadJet/p2dRingObservableDeltaPhiVsProxyPt", ringLeadJet.deltaPhi, leadJet.pt, ringLeadJet.ringObservable)                             \
  X(FOLDER "/LeadJet/p2dRingObservableDeltaThetaVsProxyPt", ringLeadJet.deltaTheta, leadJet.pt, ringLeadJet.ringObservable)                         \
  /* 1D Mass */                                                                                                            \
  X(FOLDER "/LeadJet/QA/hMass", v0LambdaLikeMass)                                                                                  \
  X(FOLDER "/LeadJet/QA/hRingObservableNumMass", v0LambdaLikeMass, ringLeadJet.ringObservable)                                                 \
  X(FOLDER "/LeadJet/hMassSigExtract", v0LambdaLikeMass)                                                                           \
  /* Counters */                                                                                                           \
  X(FOLDER "/LeadJet/QA/h2dDeltaPhiVsMass", ringLeadJet.deltaPhi, v0LambdaLikeMass)                                                         \
  X(FOLDER "/LeadJet/QA/h2dDeltaThetaVsMass", ringLeadJet.deltaTheta, v0LambdaLikeMass)                                                     \
  X(FOLDER "/LeadJet/QA/h3dDeltaPhiVsMassVsLambdaPt", ringLeadJet.deltaPhi, v0LambdaLikeMass, v0pt)                                         \
  X(FOLDER "/LeadJet/QA/h3dDeltaThetaVsMassVsLambdaPt", ringLeadJet.deltaTheta, v0LambdaLikeMass, v0pt)                                     \
  X(FOLDER "/LeadJet/QA/h3dDeltaPhiVsMassVsLeadJetPt", ringLeadJet.deltaPhi, v0LambdaLikeMass, leadJet.pt)                                \
  X(FOLDER "/LeadJet/QA/h3dDeltaThetaVsMassVsLeadJetPt", ringLeadJet.deltaTheta, v0LambdaLikeMass, leadJet.pt)                            \
  X(FOLDER "/LeadJet/QA/h3dDeltaPhiVsMassVsCent", ringLeadJet.deltaPhi, v0LambdaLikeMass, centrality)                                       \
  X(FOLDER "/LeadJet/QA/h3dDeltaThetaVsMassVsCent", ringLeadJet.deltaTheta, v0LambdaLikeMass, centrality)                                   \
  /* TProfile of Ring vs Mass */                                                                                           \
  X(FOLDER "/LeadJet/pRingObservableMass", v0LambdaLikeMass, ringLeadJet.ringObservable)                                                       \
  /* 2D Profiles: Angle vs Mass */                                                                                         \
  X(FOLDER "/LeadJet/p2dRingObservableDeltaPhiVsMass", ringLeadJet.deltaPhi, v0LambdaLikeMass, ringLeadJet.ringObservable)                              \
  X(FOLDER "/LeadJet/p2dRingObservableDeltaThetaVsMass", ringLeadJet.deltaTheta, v0LambdaLikeMass, ringLeadJet.ringObservable)                          \
  X(FOLDER "/LeadJet/p2dRingObservableEtaLambdaVsMass", v0eta, v0LambdaLikeMass, ringLeadJet.ringObservable)                                   \
  /* 2D Profiles: EtaProxy vs Mass */                                                                                      \
  X(FOLDER "/LeadJet/p2dRingObservableProxyEtaVsMass", leadJet.eta, v0LambdaLikeMass, ringLeadJet.ringObservable)                          \
  X(FOLDER "/LeadJet/h2dCounterProxyEtaVsMass", leadJet.eta, v0LambdaLikeMass)                                                 \
  /* 2D Profile: Ring vs Eta variables */                                                                                  \
  X(FOLDER "/LeadJet/EtaDependence/hCounterEtaLambdaMinusProxyEta", v0eta - leadJet.eta)                                           \
  X(FOLDER "/LeadJet/EtaDependence/pRingObservableEtaLambdaMinusProxyEta", v0eta - leadJet.eta, ringLeadJet.ringObservable)                    \
  X(FOLDER "/LeadJet/EtaDependence/p2dRingObservableEtaLambdaVsProxyEta", v0eta, leadJet.eta, ringLeadJet.ringObservable)                      \
  X(FOLDER "/LeadJet/EtaDependence/h2dCounterEtaLambdaVsProxyEta", v0eta, leadJet.eta)                                             \
  X(FOLDER "/LeadJet/EtaDependence/p2dRingObservableEtaLambdaVsProxyEta_FineBins", v0eta, leadJet.eta, ringLeadJet.ringObservable)             \
  X(FOLDER "/LeadJet/EtaDependence/h2dCounterEtaLambdaVsProxyEta_FineBins", v0eta, leadJet.eta)                                    \
  /* 3D Profiles: Angle vs Mass vs Lambda pT */                                                                            \
  X(FOLDER "/LeadJet/p3dRingObservableDeltaPhiVsMassVsLambdaPt", ringLeadJet.deltaPhi, v0LambdaLikeMass, v0pt, ringLeadJet.ringObservable)              \
  X(FOLDER "/LeadJet/p3dRingObservableDeltaThetaVsMassVsLambdaPt", ringLeadJet.deltaTheta, v0LambdaLikeMass, v0pt, ringLeadJet.ringObservable)          \
  /* 3D Profiles: Angle vs Mass vs Lead Jet pT */                                                                          \
  X(FOLDER "/LeadJet/p3dRingObservableDeltaPhiVsMassVsProxyPt", ringLeadJet.deltaPhi, v0LambdaLikeMass, leadJet.pt, ringLeadJet.ringObservable)     \
  X(FOLDER "/LeadJet/p3dRingObservableDeltaThetaVsMassVsProxyPt", ringLeadJet.deltaTheta, v0LambdaLikeMass, leadJet.pt, ringLeadJet.ringObservable) \
  /* 2D Profile: Mass vs Centrality */                                                                                     \
  X(FOLDER "/LeadJet/p2dRingObservableMassVsCent", v0LambdaLikeMass, centrality, ringLeadJet.ringObservable)                                   \
  /* 3D Profiles: Angle vs Mass vs Centrality */                                                                           \
  X(FOLDER "/LeadJet/p3dRingObservableDeltaPhiVsMassVsCent", ringLeadJet.deltaPhi, v0LambdaLikeMass, centrality, ringLeadJet.ringObservable)            \
  X(FOLDER "/LeadJet/p3dRingObservableDeltaThetaVsMassVsCent", ringLeadJet.deltaTheta, v0LambdaLikeMass, centrality, ringLeadJet.ringObservable)        \
  X(FOLDER "/LeadJet/pRingVsCentrality", centrality, ringLeadJet.ringObservable)                                                               \
  /* 2D Profiles of the ring: lab momentum planes */                                                                      \
  X(FOLDER "/LeadJet/RingMaps/p2dRingObservableVsPxPy", v0px, v0py, ringLeadJet.ringObservable)                                                        \
  X(FOLDER "/LeadJet/RingMaps/p2dRingObservableVsPzPx", v0pz, v0px, ringLeadJet.ringObservable)                                                        \
  X(FOLDER "/LeadJet/RingMaps/p2dRingObservableVsPyPz", v0py, v0pz, ringLeadJet.ringObservable)                                                        \
  /* Ring scalar on the AEE planes. R is invariant under the rotation, so only the binning should change */               \
  X(FOLDER "/LeadJet/RingMaps/p2dRingObservableVsPxAeePyAee", v0pxAee, v0pyAee, ringLeadJet.ringObservable)                                            \
  X(FOLDER "/LeadJet/RingMaps/p2dRingObservableVsPzPxAee", v0pz, v0pxAee, ringLeadJet.ringObservable)                                                  \
  X(FOLDER "/LeadJet/RingMaps/p2dRingObservableVsPyAeePz", v0pyAee, v0pz, ringLeadJet.ringObservable)                                                  \
  /* Ring scalar on the coordinates transverse to the jet (PrimeJet). */                                                  \
  X(FOLDER "/LeadJet/RingMaps/p2dRingObservableVsPxPyPrimeJet", v0pxPrimeJet, v0pyPrimeJet, ringLeadJet.ringObservable)                                \
  X(FOLDER "/LeadJet/RingMaps/p2dRingObservableVsPzPxPrimeJet", v0pzPrimeJet, v0pxPrimeJet, ringLeadJet.ringObservable)                                \
  X(FOLDER "/LeadJet/RingMaps/p2dRingObservableVsPyPzPrimeJet", v0pyPrimeJet, v0pzPrimeJet, ringLeadJet.ringObservable)                                \
  /* Ring projection kernel */                                                                                            \
  X(FOLDER "/LeadJet/RingKernel/p2dRingObservableCosDeltaThetaVsProxyZ", ringLeadJet.cosDeltaTheta, leadJetZ, ringLeadJet.ringObservable)                    \
  X(FOLDER "/LeadJet/RingKernel/h2dCountsCosDeltaThetaVsProxyZ", ringLeadJet.cosDeltaTheta, leadJetZ)                                            \
  X(FOLDER "/LeadJet/RingKernel/p3dRingObservableCosDeltaThetaVsProxyZVsLambdaZ", ringLeadJet.cosDeltaTheta, leadJetZ, lambdaZ, ringLeadJet.ringObservable)  \
  /* Jet-transverse coordinate system's polarization maps. Filled here to use existing leadJet.isValid check */        \
  X(FOLDER "/PolMaps/PrimeJet/p2dPxStarPrimeJet_vsPxPyPrimeJet", v0pxPrimeJet, v0pyPrimeJet, polStarXPrimeJet)            \
  X(FOLDER "/PolMaps/PrimeJet/p2dPyStarPrimeJet_vsPxPyPrimeJet", v0pxPrimeJet, v0pyPrimeJet, polStarYPrimeJet)            \
  X(FOLDER "/PolMaps/PrimeJet/p2dPzStarPrimeJet_vsPxPyPrimeJet", v0pxPrimeJet, v0pyPrimeJet, polStarZPrimeJet)            \
  X(FOLDER "/PolMaps/PrimeJet/h2dCountsVsPxPyPrimeJet", v0pxPrimeJet, v0pyPrimeJet)                                       \
  X(FOLDER "/PolMaps/PrimeJet/p2dPxStarPrimeJet_vsPzPxPrimeJet", v0pzPrimeJet, v0pxPrimeJet, polStarXPrimeJet)            \
  X(FOLDER "/PolMaps/PrimeJet/p2dPyStarPrimeJet_vsPzPxPrimeJet", v0pzPrimeJet, v0pxPrimeJet, polStarYPrimeJet)            \
  X(FOLDER "/PolMaps/PrimeJet/p2dPzStarPrimeJet_vsPzPxPrimeJet", v0pzPrimeJet, v0pxPrimeJet, polStarZPrimeJet)            \
  X(FOLDER "/PolMaps/PrimeJet/h2dCountsVsPzPxPrimeJet", v0pzPrimeJet, v0pxPrimeJet)                                       \
  X(FOLDER "/PolMaps/PrimeJet/p2dPxStarPrimeJet_vsPyPzPrimeJet", v0pyPrimeJet, v0pzPrimeJet, polStarXPrimeJet)            \
  X(FOLDER "/PolMaps/PrimeJet/p2dPyStarPrimeJet_vsPyPzPrimeJet", v0pyPrimeJet, v0pzPrimeJet, polStarYPrimeJet)            \
  X(FOLDER "/PolMaps/PrimeJet/p2dPzStarPrimeJet_vsPyPzPrimeJet", v0pyPrimeJet, v0pzPrimeJet, polStarZPrimeJet)            \
  X(FOLDER "/PolMaps/PrimeJet/h2dCountsVsPyPzPrimeJet", v0pyPrimeJet, v0pzPrimeJet)                                       \
  /* QAs on DeltaPhiJet -- Filed here as ringLeadJet.deltaPhi is only defined under leadJet.isValid */                          \
  X(FOLDER "/LeadJet/QA/pPxStarDeltaPhi", ringLeadJet.deltaPhi, polStarX)                                                          \
  X(FOLDER "/LeadJet/QA/pPyStarDeltaPhi", ringLeadJet.deltaPhi, polStarY)                                                          \
  X(FOLDER "/LeadJet/QA/pPzStarDeltaPhi", ringLeadJet.deltaPhi, polStarZ)                                                          \
  X(FOLDER "/LeadJet/QA/p2dPxStarDeltaPhiVsLambdaPt", ringLeadJet.deltaPhi, v0pt, polStarX)                                        \
  X(FOLDER "/LeadJet/QA/p2dPyStarDeltaPhiVsLambdaPt", ringLeadJet.deltaPhi, v0pt, polStarY)                                        \
  X(FOLDER "/LeadJet/QA/p2dPzStarDeltaPhiVsLambdaPt", ringLeadJet.deltaPhi, v0pt, polStarZ)                                        \
  /* KappaEff moments: kappa = 3 <num>/<den>, and products for the covariances */                                         \
  X(FOLDER "/LeadJet/KappaEff/pKappaNumVsMass", v0LambdaLikeMass, ringLeadJet.kappaNum)                                     \
  X(FOLDER "/LeadJet/KappaEff/pKappaDenVsMass", v0LambdaLikeMass, ringLeadJet.kappaDen)                                     \
  X(FOLDER "/LeadJet/KappaEff/pKappaNumTimesDenVsMass", v0LambdaLikeMass, ringLeadJet.kappaNum * ringLeadJet.kappaDen)               \
  X(FOLDER "/LeadJet/KappaEff/pRingTimesDenVsMass", v0LambdaLikeMass, ringLeadJet.ringObservable * ringLeadJet.kappaDen)                \
  /* CheapSigExtract mass region equivalents of KappaEff (bin 1 sideband, bin 2 peak, otherwise underflow) */             \
  X(FOLDER "/LeadJet/KappaEff/pRingVsMassRegion", massRegion, ringLeadJet.ringObservable)                                      \
  X(FOLDER "/LeadJet/KappaEff/pKappaNumVsMassRegion", massRegion, ringLeadJet.kappaNum)                                     \
  X(FOLDER "/LeadJet/KappaEff/pKappaDenVsMassRegion", massRegion, ringLeadJet.kappaDen)                                     \
  X(FOLDER "/LeadJet/KappaEff/pKappaNumTimesDenVsMassRegion", massRegion, ringLeadJet.kappaNum * ringLeadJet.kappaDen)               \
  X(FOLDER "/LeadJet/KappaEff/pRingTimesDenVsMassRegion", massRegion, ringLeadJet.ringObservable * ringLeadJet.kappaDen)                \
  /* Differential KappaEff -- vs proxy pT and vs centrality, still per mass region (axes shared with event mixing) */     \
  X(FOLDER "/LeadJet/KappaEff/Differential/p2dRingVsProxyPtVsMassRegion", leadJet.pt, massRegion, ringLeadJet.ringObservable)     \
  X(FOLDER "/LeadJet/KappaEff/Differential/p2dKappaNumVsProxyPtVsMassRegion", leadJet.pt, massRegion, ringLeadJet.kappaNum)    \
  X(FOLDER "/LeadJet/KappaEff/Differential/p2dKappaDenVsProxyPtVsMassRegion", leadJet.pt, massRegion, ringLeadJet.kappaDen)    \
  X(FOLDER "/LeadJet/KappaEff/Differential/p2dKappaNumTimesDenVsProxyPtVsMassRegion", leadJet.pt, massRegion, ringLeadJet.kappaNum * ringLeadJet.kappaDen) \
  X(FOLDER "/LeadJet/KappaEff/Differential/p2dRingTimesDenVsProxyPtVsMassRegion", leadJet.pt, massRegion, ringLeadJet.ringObservable * ringLeadJet.kappaDen) \
  X(FOLDER "/LeadJet/KappaEff/Differential/p2dRingVsCentralityVsMassRegion", centrality, massRegion, ringLeadJet.ringObservable) \
  X(FOLDER "/LeadJet/KappaEff/Differential/p2dKappaNumVsCentralityVsMassRegion", centrality, massRegion, ringLeadJet.kappaNum) \
  X(FOLDER "/LeadJet/KappaEff/Differential/p2dKappaDenVsCentralityVsMassRegion", centrality, massRegion, ringLeadJet.kappaDen) \
  X(FOLDER "/LeadJet/KappaEff/Differential/p2dKappaNumTimesDenVsCentralityVsMassRegion", centrality, massRegion, ringLeadJet.kappaNum * ringLeadJet.kappaDen) \
  X(FOLDER "/LeadJet/KappaEff/Differential/p2dRingTimesDenVsCentralityVsMassRegion", centrality, massRegion, ringLeadJet.ringObservable * ringLeadJet.kappaDen)
// (TODO: add counters for regular TH2Ds about centrality)

// For leading particle
#define RING_OBSERVABLE_LEADP_FILL_LIST(X, FOLDER)                                                            \
  X(FOLDER "/LeadP/QA/hDeltaPhi", ringLeadP.deltaPhi)                                                         \
  X(FOLDER "/LeadP/QA/hDeltaTheta", ringLeadP.deltaTheta)                                                     \
  X(FOLDER "/LeadP/QA/hProxyPt", leadP.pt)                                                                     \
  X(FOLDER "/LeadP/QA/hCosDeltaTheta", ringLeadP.cosDeltaTheta)                                               \
  /* TProfile of Ring vs Mass */                                                                              \
  X(FOLDER "/LeadP/pRingObservableMass", v0LambdaLikeMass, ringLeadP.ringObservable)                          \
  X(FOLDER "/LeadP/hRingObservableCounts", ringLeadP.ringObservable)                                          \
  X(FOLDER "/LeadP/pRingObservableDeltaPhi", ringLeadP.deltaPhi, ringLeadP.ringObservable)                         \
  X(FOLDER "/LeadP/pRingObservableDeltaTheta", ringLeadP.deltaTheta, ringLeadP.ringObservable)                     \
  X(FOLDER "/LeadP/EtaDependence/pRingObservableEtaLambda", v0eta, ringLeadP.ringObservable)                  \
  X(FOLDER "/LeadP/EtaDependence/pRingObservableProxyEta", leadP.eta, ringLeadP.ringObservable)                     \
  X(FOLDER "/LeadP/EtaDependence/pRingObservableProxyEtaHighEtaRes", leadP.eta, ringLeadP.ringObservable)           \
  X(FOLDER "/LeadP/pRingObservableIntegrated", 0., ringLeadP.ringObservable)                                  \
  X(FOLDER "/LeadP/pRingObservableLambdaPt", v0pt, ringLeadP.ringObservable)                                  \
  X(FOLDER "/LeadP/pRingObservablePVz", collisionPVz, ringLeadP.ringObservable)                               \
  X(FOLDER "/LeadP/ProxyPtDependence/pRingVsProxyPt", leadP.pt, ringLeadP.ringObservable)                           \
  X(FOLDER "/LeadP/ProxyPtDependence/p2dRingVsProxyPtVsProxyEta", leadP.pt, leadP.eta, ringLeadP.ringObservable)       \
  X(FOLDER "/LeadP/ProxyPtDependence/p2dRingVsProxyPtVsEtaV0", leadP.pt, v0eta, ringLeadP.ringObservable)             \
  X(FOLDER "/LeadP/ProxyPtDependence/p2dRingVsProxyPtVsCentrality", leadP.pt, centrality, ringLeadP.ringObservable)   \
  X(FOLDER "/LeadP/EtaDependence/p2dRingObservableEtaLambdaVsProxyEta", v0eta, leadP.eta, ringLeadP.ringObservable) \
  X(FOLDER "/LeadP/EtaDependence/h2dCounterEtaLambdaVsProxyEta", v0eta, leadP.eta)                             \
  X(FOLDER "/LeadP/RingMaps/p2dRingObservableVsPxPy", v0px, v0py, ringLeadP.ringObservable)                         \
  X(FOLDER "/LeadP/RingMaps/p2dRingObservableVsPzPx", v0pz, v0px, ringLeadP.ringObservable)                         \
  X(FOLDER "/LeadP/RingMaps/p2dRingObservableVsPyPz", v0py, v0pz, ringLeadP.ringObservable)                         \
  /* Ring projection kernel */                                                                                \
  X(FOLDER "/LeadP/RingKernel/p2dRingObservableCosDeltaThetaVsProxyZ", ringLeadP.cosDeltaTheta, leadPZ, ringLeadP.ringObservable) \
  X(FOLDER "/LeadP/RingKernel/h2dCountsCosDeltaThetaVsProxyZ", ringLeadP.cosDeltaTheta, leadPZ)               \
  X(FOLDER "/LeadP/RingKernel/p3dRingObservableCosDeltaThetaVsProxyZVsLambdaZ", ringLeadP.cosDeltaTheta, leadPZ, lambdaZ, ringLeadP.ringObservable) \
  /* 2D Profiles: EtaProxy vs Mass */                                                                         \
  X(FOLDER "/LeadP/p2dRingObservableProxyEtaVsMass", leadP.eta, v0LambdaLikeMass, ringLeadP.ringObservable)    \
  X(FOLDER "/LeadP/h2dCounterProxyEtaVsMass", leadP.eta, v0LambdaLikeMass)                                \
  /* KappaEff moments */                                                                                      \
  X(FOLDER "/LeadP/KappaEff/pKappaNumVsMass", v0LambdaLikeMass, ringLeadP.kappaNum)                           \
  X(FOLDER "/LeadP/KappaEff/pKappaDenVsMass", v0LambdaLikeMass, ringLeadP.kappaDen)                           \
  X(FOLDER "/LeadP/KappaEff/pKappaNumTimesDenVsMass", v0LambdaLikeMass, ringLeadP.kappaNum * ringLeadP.kappaDen)   \
  X(FOLDER "/LeadP/KappaEff/pRingTimesDenVsMass", v0LambdaLikeMass, ringLeadP.ringObservable * ringLeadP.kappaDen) \
  /* KappaEff with cheap signal extraction procedure */                                                       \
  X(FOLDER "/LeadP/KappaEff/pRingVsMassRegion", massRegion, ringLeadP.ringObservable)                         \
  X(FOLDER "/LeadP/KappaEff/pKappaNumVsMassRegion", massRegion, ringLeadP.kappaNum)                           \
  X(FOLDER "/LeadP/KappaEff/pKappaDenVsMassRegion", massRegion, ringLeadP.kappaDen)                           \
  X(FOLDER "/LeadP/KappaEff/pKappaNumTimesDenVsMassRegion", massRegion, ringLeadP.kappaNum * ringLeadP.kappaDen)   \
  X(FOLDER "/LeadP/KappaEff/pRingTimesDenVsMassRegion", massRegion, ringLeadP.ringObservable * ringLeadP.kappaDen) \
  /* Differential KappaEff -- vs proxy pT and vs centrality, still per mass region (axes shared with event mixing) */ \
  X(FOLDER "/LeadP/KappaEff/Differential/p2dRingVsProxyPtVsMassRegion", leadP.pt, massRegion, ringLeadP.ringObservable) \
  X(FOLDER "/LeadP/KappaEff/Differential/p2dKappaNumVsProxyPtVsMassRegion", leadP.pt, massRegion, ringLeadP.kappaNum) \
  X(FOLDER "/LeadP/KappaEff/Differential/p2dKappaDenVsProxyPtVsMassRegion", leadP.pt, massRegion, ringLeadP.kappaDen) \
  X(FOLDER "/LeadP/KappaEff/Differential/p2dKappaNumTimesDenVsProxyPtVsMassRegion", leadP.pt, massRegion, ringLeadP.kappaNum * ringLeadP.kappaDen) \
  X(FOLDER "/LeadP/KappaEff/Differential/p2dRingTimesDenVsProxyPtVsMassRegion", leadP.pt, massRegion, ringLeadP.ringObservable * ringLeadP.kappaDen) \
  X(FOLDER "/LeadP/KappaEff/Differential/p2dRingVsCentralityVsMassRegion", centrality, massRegion, ringLeadP.ringObservable) \
  X(FOLDER "/LeadP/KappaEff/Differential/p2dKappaNumVsCentralityVsMassRegion", centrality, massRegion, ringLeadP.kappaNum) \
  X(FOLDER "/LeadP/KappaEff/Differential/p2dKappaDenVsCentralityVsMassRegion", centrality, massRegion, ringLeadP.kappaDen) \
  X(FOLDER "/LeadP/KappaEff/Differential/p2dKappaNumTimesDenVsCentralityVsMassRegion", centrality, massRegion, ringLeadP.kappaNum * ringLeadP.kappaDen) \
  X(FOLDER "/LeadP/KappaEff/Differential/p2dRingTimesDenVsCentralityVsMassRegion", centrality, massRegion, ringLeadP.ringObservable * ringLeadP.kappaDen)

// A macro that encapsulates all eta checks for leading particle and V0s, along with the fills
// Parameters:
//   FOLDER       -- histogram folder string (compile-time literal)
//   LEADP_IS_POS -- bool: leadPEtaPos
//   V0_IS_POS    -- bool: lambdaEtaPos
#define RING_OBSERVABLE_LEADP_ETA_SPLIT_FILL_LIST(FOLDER, LEADP_IS_POS, V0_IS_POS)                                      \
  do {                                                                                                                  \
    if (LEADP_IS_POS) {                                                                                                 \
      /* leadP marginal: positive side */                                                                               \
      APPLY_HISTO_FILL(FOLDER "/LeadP/ProxyPtDependence/pRingVsProxyPt_PosProxyEta", leadP.pt, ringLeadP.ringObservable)            \
      if (V0_IS_POS) {                                                                                                  \
        /* V0 marginal: positive side */                                                                                \
        APPLY_HISTO_FILL(FOLDER "/LeadP/ProxyPtDependence/pRingVsProxyPt_PosEtaV0", leadP.pt, ringLeadP.ringObservable)             \
        /* Joint: (+leadP, +V0) */                                                                                      \
        APPLY_HISTO_FILL(FOLDER "/LeadP/ProxyPtDependence/pRingVsProxyPt_PosProxyEta_PosEtaV0", leadP.pt, ringLeadP.ringObservable) \
      } else {                                                                                                          \
        /* V0 marginal: negative side */                                                                                \
        APPLY_HISTO_FILL(FOLDER "/LeadP/ProxyPtDependence/pRingVsProxyPt_NegEtaV0", leadP.pt, ringLeadP.ringObservable)             \
        /* Joint: (+leadP, -V0) */                                                                                      \
        APPLY_HISTO_FILL(FOLDER "/LeadP/ProxyPtDependence/pRingVsProxyPt_PosProxyEta_NegEtaV0", leadP.pt, ringLeadP.ringObservable) \
      }                                                                                                                 \
    } else {                                                                                                            \
      /* leadP marginal: negative side */                                                                               \
      APPLY_HISTO_FILL(FOLDER "/LeadP/ProxyPtDependence/pRingVsProxyPt_NegProxyEta", leadP.pt, ringLeadP.ringObservable)            \
      if (V0_IS_POS) {                                                                                                  \
        /* V0 marginal: positive side */                                                                                \
        APPLY_HISTO_FILL(FOLDER "/LeadP/ProxyPtDependence/pRingVsProxyPt_PosEtaV0", leadP.pt, ringLeadP.ringObservable)             \
        /* Joint: (-leadP, +V0) */                                                                                      \
        APPLY_HISTO_FILL(FOLDER "/LeadP/ProxyPtDependence/pRingVsProxyPt_NegProxyEta_PosEtaV0", leadP.pt, ringLeadP.ringObservable) \
      } else {                                                                                                          \
        /* V0 marginal: negative side */                                                                                \
        APPLY_HISTO_FILL(FOLDER "/LeadP/ProxyPtDependence/pRingVsProxyPt_NegEtaV0", leadP.pt, ringLeadP.ringObservable)             \
        /* Joint: (-leadP, -V0) */                                                                                      \
        APPLY_HISTO_FILL(FOLDER "/LeadP/ProxyPtDependence/pRingVsProxyPt_NegProxyEta_NegEtaV0", leadP.pt, ringLeadP.ringObservable) \
      }                                                                                                                 \
    }                                                                                                                   \
  } while (0)

// For subleading jet:
#define RING_OBSERVABLE_2NDJET_FILL_LIST(X, FOLDER)                                                                         \
  X(FOLDER "/SubJet/QA/hDeltaPhi", ringSubJet.deltaPhi)                                                                    \
  X(FOLDER "/SubJet/QA/hDeltaTheta", ringSubJet.deltaTheta)                                                                \
  X(FOLDER "/SubJet/QA/hCosDeltaTheta", ringSubJet.cosDeltaTheta)                                                          \
  X(FOLDER "/SubJet/QA/hProxyPt", subJet.pt)                                                                         \
  /* TProfile of Ring vs Mass */                                                                                            \
  X(FOLDER "/SubJet/pRingObservableMass", v0LambdaLikeMass, ringSubJet.ringObservable)                                     \
  X(FOLDER "/SubJet/hRingObservableCounts", ringSubJet.ringObservable)                                                    \
  X(FOLDER "/SubJet/pRingObservableDeltaPhi", ringSubJet.deltaPhi, ringSubJet.ringObservable)                                   \
  X(FOLDER "/SubJet/pRingObservableDeltaTheta", ringSubJet.deltaTheta, ringSubJet.ringObservable)                               \
  X(FOLDER "/SubJet/EtaDependence/pRingObservableEtaLambda", v0eta, ringSubJet.ringObservable)                             \
  X(FOLDER "/SubJet/EtaDependence/pRingObservableProxyEta", subJet.eta, ringSubJet.ringObservable)                        \
  X(FOLDER "/SubJet/pRingObservableIntegrated", 0., ringSubJet.ringObservable)                                             \
  X(FOLDER "/SubJet/pRingObservableLambdaPt", v0pt, ringSubJet.ringObservable)                                             \
  X(FOLDER "/SubJet/pRingObservablePVz", collisionPVz, ringSubJet.ringObservable)                                         \
  X(FOLDER "/SubJet/ProxyPtDependence/pRingVsProxyPt", subJet.pt, ringSubJet.ringObservable)                              \
  X(FOLDER "/SubJet/ProxyPtDependence/p2dRingVsProxyPtVsProxyEta", subJet.pt, subJet.eta, ringSubJet.ringObservable) \
  X(FOLDER "/SubJet/ProxyPtDependence/p2dRingVsProxyPtVsEtaV0", subJet.pt, v0eta, ringSubJet.ringObservable)                \
  X(FOLDER "/SubJet/ProxyPtDependence/p2dRingVsProxyPtVsCentrality", subJet.pt, centrality, ringSubJet.ringObservable)      \
  X(FOLDER "/SubJet/EtaDependence/p2dRingObservableEtaLambdaVsProxyEta", v0eta, subJet.eta, ringSubJet.ringObservable)    \
  X(FOLDER "/SubJet/EtaDependence/h2dCounterEtaLambdaVsProxyEta", v0eta, subJet.eta)                                 \
  /* 2D Profiles: EtaProxy vs Mass */                                                                                       \
  X(FOLDER "/SubJet/p2dRingObservableProxyEtaVsMass", subJet.eta, v0LambdaLikeMass, ringSubJet.ringObservable)      \
  X(FOLDER "/SubJet/h2dCounterProxyEtaVsMass", subJet.eta, v0LambdaLikeMass)                                   \
  /* KappaEff moments */                                                                                                    \
  X(FOLDER "/SubJet/KappaEff/pKappaNumVsMass", v0LambdaLikeMass, ringSubJet.kappaNum)                                      \
  X(FOLDER "/SubJet/KappaEff/pKappaDenVsMass", v0LambdaLikeMass, ringSubJet.kappaDen)                                      \
  X(FOLDER "/SubJet/KappaEff/pKappaNumTimesDenVsMass", v0LambdaLikeMass, ringSubJet.kappaNum * ringSubJet.kappaDen)             \
  X(FOLDER "/SubJet/KappaEff/pRingTimesDenVsMass", v0LambdaLikeMass, ringSubJet.ringObservable * ringSubJet.kappaDen)           \
  /* KappaEff with cheap signal extraction procedure */                                                                     \
  X(FOLDER "/SubJet/KappaEff/pRingVsMassRegion", massRegion, ringSubJet.ringObservable)                                    \
  X(FOLDER "/SubJet/KappaEff/pKappaNumVsMassRegion", massRegion, ringSubJet.kappaNum)                                      \
  X(FOLDER "/SubJet/KappaEff/pKappaDenVsMassRegion", massRegion, ringSubJet.kappaDen)                                      \
  X(FOLDER "/SubJet/KappaEff/pKappaNumTimesDenVsMassRegion", massRegion, ringSubJet.kappaNum * ringSubJet.kappaDen)             \
  X(FOLDER "/SubJet/KappaEff/pRingTimesDenVsMassRegion", massRegion, ringSubJet.ringObservable * ringSubJet.kappaDen)           \
  /* Differential KappaEff -- vs proxy pT and vs centrality, still per mass region (axes shared with event mixing) */       \
  X(FOLDER "/SubJet/KappaEff/Differential/p2dRingVsProxyPtVsMassRegion", subJet.pt, massRegion, ringSubJet.ringObservable) \
  X(FOLDER "/SubJet/KappaEff/Differential/p2dKappaNumVsProxyPtVsMassRegion", subJet.pt, massRegion, ringSubJet.kappaNum)  \
  X(FOLDER "/SubJet/KappaEff/Differential/p2dKappaDenVsProxyPtVsMassRegion", subJet.pt, massRegion, ringSubJet.kappaDen)  \
  X(FOLDER "/SubJet/KappaEff/Differential/p2dKappaNumTimesDenVsProxyPtVsMassRegion", subJet.pt, massRegion, ringSubJet.kappaNum * ringSubJet.kappaDen) \
  X(FOLDER "/SubJet/KappaEff/Differential/p2dRingTimesDenVsProxyPtVsMassRegion", subJet.pt, massRegion, ringSubJet.ringObservable * ringSubJet.kappaDen) \
  X(FOLDER "/SubJet/KappaEff/Differential/p2dRingVsCentralityVsMassRegion", centrality, massRegion, ringSubJet.ringObservable) \
  X(FOLDER "/SubJet/KappaEff/Differential/p2dKappaNumVsCentralityVsMassRegion", centrality, massRegion, ringSubJet.kappaNum) \
  X(FOLDER "/SubJet/KappaEff/Differential/p2dKappaDenVsCentralityVsMassRegion", centrality, massRegion, ringSubJet.kappaDen) \
  X(FOLDER "/SubJet/KappaEff/Differential/p2dKappaNumTimesDenVsCentralityVsMassRegion", centrality, massRegion, ringSubJet.kappaNum * ringSubJet.kappaDen) \
  X(FOLDER "/SubJet/KappaEff/Differential/p2dRingTimesDenVsCentralityVsMassRegion", centrality, massRegion, ringSubJet.ringObservable * ringSubJet.kappaDen)

#define POLARIZATION_PROFILE_FILL_LIST(X, FOLDER)                          \
  /* 1D TProfiles vs v0phi */                                              \
  X(FOLDER "/QA/pPxStarPhi", v0phiToFillHists, polStarX)                   \
  X(FOLDER "/QA/pPyStarPhi", v0phiToFillHists, polStarY)                   \
  X(FOLDER "/QA/pPzStarPhi", v0phiToFillHists, polStarZ)                   \
  /* 1D TProfiles vs the AEE angle */                                               \
  X(FOLDER "/QA/pPxStarPhiAEE", deltaPhiLambdaProtonStar, polStarX) \
  X(FOLDER "/QA/pPyStarPhiAEE", deltaPhiLambdaProtonStar, polStarY) \
  X(FOLDER "/QA/pPzStarPhiAEE", deltaPhiLambdaProtonStar, polStarZ) \
  /* Lab-frame momentum-plane maps */                                               \
  X(FOLDER "/PolMaps/Lab/p2dPxStar_vsPxPy", v0px, v0py, polStarX)                   \
  X(FOLDER "/PolMaps/Lab/p2dPyStar_vsPxPy", v0px, v0py, polStarY)                   \
  X(FOLDER "/PolMaps/Lab/p2dPzStar_vsPxPy", v0px, v0py, polStarZ)                   \
  X(FOLDER "/PolMaps/Lab/h2dCountsVsPxPy", v0px, v0py)                              \
  X(FOLDER "/PolMaps/Lab/p2dPxStar_vsPzPx", v0pz, v0px, polStarX)                   \
  X(FOLDER "/PolMaps/Lab/p2dPyStar_vsPzPx", v0pz, v0px, polStarY)                   \
  X(FOLDER "/PolMaps/Lab/p2dPzStar_vsPzPx", v0pz, v0px, polStarZ)                   \
  X(FOLDER "/PolMaps/Lab/h2dCountsVsPzPx", v0pz, v0px)                              \
  X(FOLDER "/PolMaps/Lab/p2dPxStar_vsPyPz", v0py, v0pz, polStarX)                   \
  X(FOLDER "/PolMaps/Lab/p2dPyStar_vsPyPz", v0py, v0pz, polStarY)                   \
  X(FOLDER "/PolMaps/Lab/p2dPzStar_vsPyPz", v0py, v0pz, polStarZ)                   \
  X(FOLDER "/PolMaps/Lab/h2dCountsVsPyPz", v0py, v0pz)                              \
  /* Aee frame: acceptance maps (radius = pT^Lambda, azimuth = PhiAEE) */           \
  X(FOLDER "/PolMaps/Aee/p2dPtStar_vsPxAeePyAee", v0pxAee, v0pyAee, polStarTAee)    \
  X(FOLDER "/PolMaps/Aee/p2dPzStar_vsPxAeePyAee", v0pxAee, v0pyAee, polStarZ)       \
  X(FOLDER "/PolMaps/Aee/h2dCountsVsPxAeePyAee", v0pxAee, v0pyAee)                  \
  X(FOLDER "/PolMaps/Aee/p2dPtStar_vsPzPxAee", v0pz, v0pxAee, polStarTAee)          \
  X(FOLDER "/PolMaps/Aee/p2dPzStar_vsPzPxAee", v0pz, v0pxAee, polStarZ)             \
  X(FOLDER "/PolMaps/Aee/h2dCountsVsPzPxAee", v0pz, v0pxAee)                        \
  X(FOLDER "/PolMaps/Aee/p2dPtStar_vsPyAeePz", v0pyAee, v0pz, polStarTAee)          \
  X(FOLDER "/PolMaps/Aee/p2dPzStar_vsPyAeePz", v0pyAee, v0pz, polStarZ)             \
  X(FOLDER "/PolMaps/Aee/h2dCountsVsPyAeePz", v0pyAee, v0pz)                        \
  /* PrimeV0 frame: the Lambda production plane (a null test) */                    \
  X(FOLDER "/PolMaps/PrimeV0/p2dPxStarPrimeV0_vsPzPt", v0pz, v0pt, polStarXPrimeV0) \
  X(FOLDER "/PolMaps/PrimeV0/p2dPyStarPrimeV0_vsPzPt", v0pz, v0pt, polStarYPrimeV0) \
  X(FOLDER "/PolMaps/PrimeV0/p2dPzStar_vsPzPt", v0pz, v0pt, polStarZ)               \
  X(FOLDER "/PolMaps/PrimeV0/h2dCountsVsPzPt", v0pz, v0pt)

// Apply the macros (notice I had to include the semicolon (";") after the function, so you don't need to
// write that when calling this APPLY_HISTO_FILL. The code will look weird, but without this the compiler
// would not know to end each statement with a semicolon):
#define APPLY_HISTO_FILL(NAME, ...) histosRingFamily.fill(HIST(NAME), __VA_ARGS__);

// Delta Method Fill Lists
#define DELTA_INTEGRATED_FILL_LIST(X, FOLDER, r, n)              \
  X(FOLDER "/DeltaMethod/hIntegrated", 0.5, r)                   \
  X(FOLDER "/DeltaMethod/hIntegrated", 1.5, (double)(n))         \
  X(FOLDER "/DeltaMethod/hIntegrated", 2.5, (r) * (r))           \
  X(FOLDER "/DeltaMethod/hIntegrated", 3.5, (double)((n) * (n))) \
  X(FOLDER "/DeltaMethod/hIntegrated", 4.5, (r) * (n))

#define DELTA_2D_FILL_LIST(X, FOLDER, HIST_NAME, center, r, n)          \
  X(FOLDER "/DeltaMethod/" HIST_NAME, center, 0.5, r)                   \
  X(FOLDER "/DeltaMethod/" HIST_NAME, center, 1.5, (double)(n))         \
  X(FOLDER "/DeltaMethod/" HIST_NAME, center, 2.5, (r) * (r))           \
  X(FOLDER "/DeltaMethod/" HIST_NAME, center, 3.5, (double)((n) * (n))) \
  X(FOLDER "/DeltaMethod/" HIST_NAME, center, 4.5, (r) * (n))

// Master flush macro to dump an event tracker into the histograms:
#define FLUSH_DELTA_TRACKER(FOLDER, TRACKER, AXIS_PT, AXIS_MASS, AXIS_DTHETA)                    \
  if ((TRACKER).nInt > 0) {                                                                      \
    DELTA_INTEGRATED_FILL_LIST(APPLY_HISTO_FILL, FOLDER, (TRACKER).rInt, (TRACKER).nInt)         \
  }                                                                                              \
  for (size_t bin = 0; bin < (TRACKER).rPt.size(); ++bin) {                                      \
    int nVal = (TRACKER).nPt[bin];                                                               \
    if (nVal == 0)                                                                               \
      continue;                                                                                  \
    double rVal = (TRACKER).rPt[bin];                                                            \
    double center = (AXIS_PT)->GetBinCenter(bin);                                                \
    DELTA_2D_FILL_LIST(APPLY_HISTO_FILL, FOLDER, "h2dLambdaPtVsDeltaComp", center, rVal, nVal)   \
  }                                                                                              \
  for (size_t bin = 0; bin < (TRACKER).rMass.size(); ++bin) {                                    \
    int nVal = (TRACKER).nMass[bin];                                                             \
    if (nVal == 0)                                                                               \
      continue;                                                                                  \
    double rVal = (TRACKER).rMass[bin];                                                          \
    double center = (AXIS_MASS)->GetBinCenter(bin);                                              \
    DELTA_2D_FILL_LIST(APPLY_HISTO_FILL, FOLDER, "h2dMassVsDeltaComp", center, rVal, nVal)       \
  }                                                                                              \
  for (size_t bin = 0; bin < (TRACKER).rDtheta.size(); ++bin) {                                  \
    int nVal = (TRACKER).nDtheta[bin];                                                           \
    if (nVal == 0)                                                                               \
      continue;                                                                                  \
    double rVal = (TRACKER).rDtheta[bin];                                                        \
    double center = (AXIS_DTHETA)->GetBinCenter(bin);                                            \
    DELTA_2D_FILL_LIST(APPLY_HISTO_FILL, FOLDER, "h2dDeltaThetaVsDeltaComp", center, rVal, nVal) \
  }

struct lambdajetpolarizationionsderived {
  // Define histogram registries:
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};
  HistogramRegistry histosRingFamily{"HistosRingFamily", {}, OutputObjHandlingPolicy::AnalysisObject};

  // Master analysis switches:
  Configurable<bool> analyseLambda{"analyseLambda", true, "process Lambda-like candidates"};
  Configurable<bool> analyseAntiLambda{"analyseAntiLambda", false, "process AntiLambda-like candidates"};
  Configurable<bool> analyseMagField{"analyseMagField", true, "analyse efficiency effects wrt magnetic field"}; // DerivedData lacks actual magField, so this is only useful for runs with only one field polarity
  Configurable<bool> useRingZ{"useRingZ", false, "redefine the ring as the projection R_z = P_z cdot n_z for every proxy"};
  Configurable<bool> useRingZT{"useRingZT", false, "redefine the ring as R_z^T = R_z - 'helicity term'"}; // Exclusive with useRingZ (they cannot both be active!)
  // Configurable<bool> doPPAnalysis{"doPPAnalysis", false, "if in pp, set to true. Default is HI"};

  // A very inexpensive "signal extraction" imitation based on v0InMassPeak bool:
  // (Uses a mass interval to remove or include V0s from the final AnalysisResults to take advantage of existing post-processing codes)
  Configurable<bool> excludeOutOfPeakQA{"excludeOutOfPeakQA", false, "removes all V0s outside an approximate +/- PeakWindowNSigma*sigma window from the mass peak"}; // A naive estimator of signal
  Configurable<bool> excludeInPeakQA{"excludeInPeakQA", false, "uses only the V0s outside an approximate +/- (SidebandInnerNSigma, SidebandOuterNSigma)*sigma window from the mass peak. Should be the same width as PeakWindowNSigma."}; // A naive estimator of background
  Configurable<float> PeakWindowNSigma{"PeakWindowNSigma", 1.5f, "Size for peak window, centered in LambdaMass from the PDG."};
  Configurable<float> SidebandInnerNSigma{"SidebandInnerNSigma", 5.5f, "Absolute value for lower end of sideband window."};
  Configurable<float> SidebandOuterNSigma{"SidebandOuterNSigma", 7.0f, "Absolute value for upper end of sideband window."};

  // Per-family histogram switches:
  // (Each family books >100 histograms, so it is necessary to keep some of these switches off to avoid the HistogramRegistry limit)
  struct : ConfigurableGroup {
    std::string prefix = "familySwitches"; // JSON group name
    Configurable<bool> doFamilyRing{"doFamilyRing", true, "Book and fill the 'Ring' family (no additional cuts). Keep this on for most passes."};
    Configurable<bool> doFamilyRingKinematicCuts{"doFamilyRingKinematicCuts", false, "Book and fill the 'RingKinematicCuts' family (Lambda kinematic cuts applied)."};
    Configurable<bool> doFamilyJetKinematicCuts{"doFamilyJetKinematicCuts", true, "Book and fill the 'JetKinematicCuts' family (jet kinematic cuts applied)."};
    Configurable<bool> doFamilyJetAndLambdaKinematicCuts{"doFamilyJetAndLambdaKinematicCuts", false, "Book and fill the 'JetAndLambdaKinematicCuts' family (both cuts applied)."};
  } familySwitches;

  // QA switches:
  struct : ConfigurableGroup {
    std::string prefix = "qaSwitches";                                                                                                           // JSON group name
    Configurable<bool> doFakePolDiagnosticsQA{"doFakePolDiagnosticsQA", true, "Book and fill the EtaStudy/ and HelicityEfficiencyQA/ folders."}; // The largest per-V0 fill cost in this task
    Configurable<bool> doEventMixingQA{"doEventMixingQA", true, "Book and fill the EventMixingQA/ folder. Requires fakePolSwitches.doMixedEventProxies."};
  } qaSwitches;

  // Centrality:
  Configurable<int> centralityEstimator{"centralityEstimator", kCentFT0M, "Run 3 centrality estimator (0:CentFT0C, 1:CentFT0M, 2:CentFV0A)"}; // Default is FT0M
  Configurable<float> maxZVtxPosition{"maxZVtxPosition", 5., "max Z vtx position [cm]"};                                                     // An additional post-processing cut after derived data was written. Same default as lambdaJetPolarizationIons.cxx producer

  // QAs that purposefully "break" the analysis
  // -- All of these tests should give us zero signal if the source is truly Lambda Polarization from vortices
  struct : ConfigurableGroup {
    std::string prefix = "fakePolSwitches"; // JSON group name
    Configurable<bool> forcePolSignQA{"forcePolSignQA", false, "force antiLambda decay constant to be positive: should kill all the signal, if any. For QA"};
    Configurable<bool> forcePerpToJet{"forcePerpToJet", false, "force jet direction to be perpendicular to jet estimator"};
    Configurable<bool> forceJetDirectionSmudge{"forceJetDirectionSmudge", false, "fluctuate jet direction by 10% of R around original axis. For QA (tests sensibility)"};
    Configurable<bool> forceRandJet{"forceRandJet", false, "makes jet direction random. A QA for AEE fake signal and its removal"};
    Configurable<bool> forcePreviousJet{"forcePreviousJet", false, "uses previous event's jet direction instead of a random sample. A baseline for fake signal removal"};
    Configurable<bool> forceDatalikeJet{"forceDatalikeJet", false, "a compromise between forceRandJet and forcePreviousJet. Parameterized distribution from data"};
    Configurable<bool> doMixedEventProxies{"doMixedEventProxies", false, "mix leadP/leadJet/subJet directions between events using (proxy pt, Zvtx, centrality) bins -- three independent mixings, one per proxy"};
    Configurable<int> mixedEventWindowSize{"mixedEventWindowSize", 20, "number of neighbours for doMixedEventProxies: how many similar collisions stay eligible as mixing partners at once."}; // Should not be much higher than 30 for pp jets
    Configurable<int32_t> mixingIdxGapSize{"mixingIdxGapSize", 0, "Reject a mixing partner with collision index within +/- mixingIdxGapSize rows of the current collision. 0 disables the cut."}; // Continuous-readout cannibalization test.
    Configurable<bool> gatePtOnArtificialProxies{"gatePtOnArtificialProxies", false, "Apply the minimum-pT selection to artificial proxies (PerpToJet/DirectionSmudge/RandJet/DatalikeJet) after their pT is recomputed."}; //  Off by default, but the Pt cut's biasing can be measured on demand. Borrowed physical proxies always pass it by construction.
    Configurable<bool> gateEtaOnArtificialProxies{"gateEtaOnArtificialProxies", true, "Apply an |eta| acceptance cut to artificial proxies (PerpToJet/DirectionSmudge/RandJet/DatalikeJet)"}; // forceRandJet samples the full 4pi, so without this it produces proxies far outside acceptance.
    Configurable<float> maxLeadPProxyEta{"maxLeadPProxyEta", 0.9f, "|eta| ceiling for artificial leading-particle proxies."};
    Configurable<float> maxJetProxyEta{"maxJetProxyEta", 0.5f, "|eta| ceiling for artificial lead jet sublead jet proxies."};
    Configurable<int> nProxyResamples{"nProxyResamples", 1, "The amount of resamplings of jet direction per event. Use ONLY for forceRandJet, forceDatalikeJet and doMixedEventProxies"};
  } fakePolSwitches;

  Configurable<float> jetR{"jetR", 0.4f, "Radius of the jet"}; // Provide manually, please. Also used for forceJetDirectionSmudge
  Configurable<float> minLeadParticlePt{"minLeadParticlePt", 4.0f, "Minimum Pt for a lead track to be considered a valid proxy for a jet (may be more restrictive than TableProducer)"};
  Configurable<float> minLeadJetPt{"minLeadJetPt", 10.0f, "Minimum Pt for leading jet to be considered valid (may be more restrictive than TableProducer)"};
  Configurable<float> minSubLeadJetPt{"minSubLeadJetPt", 8.0f, "Minimum Pt for subleading jet to be considered valid (may be more restrictive than TableProducer)"};

  struct : ConfigurableGroup {
    std::string prefix = "analysisLevelCuts"; // JSON group name
    Configurable<bool> doAnalysisLevelCuts{"doAnalysisLevelCuts", false, "Perform topologic, kinematic and PID cuts on derived data. Useful for systematics."};
    // Kinematic cuts:
    Configurable<float> v0MinPt{"v0MinPt", 0.3f, "Minimum Pt for V0. Phenomenology suggests 0.5 GeV/c."};
    Configurable<float> v0MaxPt{"v0MaxPt", 4.f, "Maximum Pt for V0. Phenomenology suggests 1.5 GeV/c."};
    Configurable<float> v0MaxRap{"v0MaxRap", 0.5f, "Rapidity cut for V0. Phenomenology suggests |y| < 0.5."}; // For pT~0.3, |y|<0.5 means |etaLambda|<~1.45
    Configurable<float> v0MaxEta{"v0MaxEta", 999.f, "Pseudorapidity cut for V0. Complementary to the v0MaxRap cut, closer to detector-related effects."};
    // Armenteros cuts to remove K0s (35% of sample) and possible photons (<1% sample, but cheap to remove):
    Configurable<float> apAlphaMin{"apAlphaMin", 0.4f, "Armenteros-Podolanski min #alpha cut."};
    Configurable<float> apAlphaMax{"apAlphaMax", 0.95f, "Armenteros-Podolanski max #alpha cut."};
    Configurable<float> apQtMin{"apQtMin", 0.008f, "Armenteros-Podolanski min q_{T} cut."};
    Configurable<float> apQtMax{"apQtMax", 0.11f, "Armenteros-Podolanski max q_{T} cut."};
    // TPC-related:
    Configurable<float> nSigmaTPCPrLike{"nSigmaTPCPrLike", 4.f, "TableProducer default is 5."}; // Same as tpcPidNsigmaCut from TableProducer
    Configurable<float> nSigmaTPCPiLike{"nSigmaTPCPiLike", 4.f, "TableProducer default is 5."};
    // Topological cuts:
    Configurable<float> v0MaxDcaDau{"v0MaxDcaDau", 999.f, "Max DCA between V0 daughters (cm). TableProducer default is 1.2."};
    Configurable<float> v0MinCosPA{"v0MinCosPA", -1.f, "Min V0 cosine of pointing angle. TableProducer default is 0.995."};
    Configurable<float> v0MinRadius{"v0MinRadius", -1.f, "Min V0 decay radius (cm). TableProducer default is 1.0."};
    Configurable<float> v0MaxRadius{"v0MaxRadius", 999.f, "Max V0 decay radius (cm). TableProducer default is 1E5."};
    Configurable<float> v0MinDcaPrLikeToPV{"v0MinDcaPrLikeToPV", -1.f, "Min |DCA|_{xy} of the proton-like daughter to the PV (cm)."}; // 0.05 is builder's minimum
    Configurable<float> v0MinDcaPiLikeToPV{"v0MinDcaPiLikeToPV", -1.f, "Min |DCA|_{xy} of the pion-like daughter to the PV (cm)."};
    // Jets-related (kinematic, quenching, etc.):
    // (TODO)
    // An additional V0-level cut on lab phi (currently for testing only):
    Configurable<std::vector<float>> v0PhiLimits{"v0PhiLimits", {0.f, constants::math::TwoPI}, "Limits on the V0 phi, in [0,2pi]. Testing only."};

    // V0 Mass cuts to remove candidates outside of the usual signal extraction limits (a cheap QA procedure before full signal extraction):
    Configurable<float> v0MassMin{"v0MassMin", 1.09f, "V0 mass min cut. Removes V0s from combinatoric background, for QA"}; // Inert cut is -999.f
    Configurable<float> v0MassMax{"v0MassMax", 1.14f, "V0 mass max cut. Removes V0s from combinatoric background, for QA"}; // Inert cut is +999.f
  } analysisLevelCuts;

  /////////////////////////
  // Configurable blocks:
  // Histogram axes configuration:
  struct : ConfigurableGroup {
    std::string prefix = "axisConfigurations"; // JSON group name
    ConfigurableAxis axisPt{"axisPt", {VARIABLE_WIDTH, 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.2f, 2.4f, 2.6f, 2.8f, 3.0f, 3.2f, 3.4f, 3.6f, 3.8f, 4.0f, 4.4f, 4.8f, 5.2f, 5.6f, 6.0f, 6.5f, 7.0f, 7.5f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 17.0f, 19.0f, 21.0f, 23.0f, 25.0f, 30.0f, 35.0f, 40.0f, 50.0f}, "pt axis for analysis"};
    ConfigurableAxis axisPtCoarseQA{"axisPtCoarseQA", {VARIABLE_WIDTH, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 7.0f, 10.0f, 15.0f}, "pt axis for QA"};
    ConfigurableAxis axisLambdaMass{"axisLambdaMass", {450, 1.08f, 1.15f}, "#Lambda mass in GeV/c"}; // Default is {200, 1.101f, 1.131f}

    // Symmetric momentum-plane axes for the vector-field / ring 2D profiles:
    ConfigurableAxis axisLambdaPz{"axisLambdaPz", {40, -4.0f, 4.0f}, "#Lambda p_{z} (GeV/c)"};

    // Rotated-frame momentum-plane axes (Aee, PrimeV0 and PrimeJet systems):
    ConfigurableAxis axisLambdaPRot{"axisLambdaPRot", {40, -3.0f, 3.0f}, "#Lambda rotated p component (GeV/c)"};
    // pT axis for the PrimeV0 production-plane map. Matching axisLambdaPz's bin width:
    ConfigurableAxis axisLambdaPtRot{"axisLambdaPtRot", {15, 0.0f, 3.0f}, "#Lambda p_{T} (GeV/c)"};

    // Event properties:
    ConfigurableAxis axisPVz{"axisPVz", {60, -10.0f, +10.0f}, "Primary Vertex Z [cm]"};
    ConfigurableAxis axisPVzCoarse{"axisPVzCoarse", {10, -10.0f, +10.0f}, "Primary Vertex Z [cm]"}; // For RingObservable calculation, not for event mixing

    // Jet axes:
    // ConfigurableAxis axisLeadingParticlePt{"axisLeadingParticlePt", {100, 0.f, 200.f}, "Leading particle p_{T} (GeV/c)"}; // Simpler version!
    // ConfigurableAxis axisJetPt{"axisJetPt", {50, 0.f, 200.f}, "Jet p_{t} (GeV)"};
    ConfigurableAxis axisJetPt{
      "axisJetPt",
      {VARIABLE_WIDTH,
      //  0, 2, 4, 6, 8, 10, // 2 GeV bins
      //  15, 20,            // 5 GeV bins
      //  30, 40,            // 10 GeV bins
      //  60, 80,            // 20 GeV bins
      //  120, 160, 200},    // 40 GeV bins
      0, 2, 4, 6, 8, 10, // 2 GeV bins
      20, 40, 100, 200},
      "Jet p_{T} (GeV)"};
    ConfigurableAxis axisJetPtSigExtract{"axisJetPtSigExtract", {VARIABLE_WIDTH, 0, 5, 10, 12, 16, 20, 25, 30, 35, 40, 60, 100, 200}, "Jet p_{t} (GeV)"};
    ConfigurableAxis axisEta{"axisEta", {50, -1.0f, 1.0f}, "#eta"};
    ConfigurableAxis axisEtaCoarse{"axisEtaCoarse", {20, -0.9f, 0.9f}, "#eta coarse axis"};
    ConfigurableAxis axisEtaSigExtract{"axisEtaSigExtract", {10, -0.9f, 0.9f}, "#eta coarser axis (sigExtract)"};
    // ConfigurableAxis axisEtaFine{"axisEtaFine", {100, -0.9f, 0.9f}, "#eta fine axis"};
    ConfigurableAxis axisV0Eta{"axisV0Eta", {75, -1.5f, 1.5f}, "V0 #eta"}; // An axis for V0 eta, which can go up to 1.5 given standard producer selections
    ConfigurableAxis axisV0EtaCoarse{"axisV0EtaCoarse", {30, -1.5f, 1.5f}, "V0 #eta coarse"};
    ConfigurableAxis axisDeltaEtaCoarse{"axisDeltaEtaCoarse", {40, -1.8f, 1.8f}, "#Delta#eta coarse axis"};
    ConfigurableAxis axisDeltaTheta{"axisDeltaTheta", {40, 0, constants::math::PI}, "#Delta #theta_{jet}"};
    ConfigurableAxis axisCosTheta{"axisCosTheta", {50, -1, 1}, "cos(#theta)"};
    ConfigurableAxis axisCosThetaCoarse{"axisCosThetaCoarse", {10, -1, 1}, "cos(#theta)"};
    // Direction cosines. t_z = cos(theta) = tanh(eta), so these are the z components of the
    // corresponding unit vectors. Ranges follow each proxy's fiducial acceptance:
    // jets are capped at |eta| < 0.9 - R = 0.5, leading particles at |eta| < 0.9.
    ConfigurableAxis axisProxyZ{"axisProxyZ", {40, -1, 1}, "#hat{t}_{z}"};
    ConfigurableAxis axisJetZ{"axisJetZ", {40, -0.5, 0.5}, "#hat{t}_{z}"};
    ConfigurableAxis axisLeadPZ{"axisLeadPZ", {40, -0.75, 0.75}, "#hat{t}_{z}^{LeadP}"};
    ConfigurableAxis axisLambdaZ{"axisLambdaZ", {14, -0.7, 0.7}, "cos#theta_{#Lambda}"};
    ConfigurableAxis axisPhi{"axisPhi", {40, 0., constants::math::TwoPI}, "#varphi"};
    ConfigurableAxis axisDeltaPhi{"axisDeltaPhi", {40, -constants::math::PI, constants::math::PI}, "#Delta #phi_{jet}"};
    ConfigurableAxis axisDeltaPhiCoarse{"axisDeltaPhiCoarse", {32, -constants::math::PI, constants::math::PI}, "#Delta #phi coarse"}; // (signal extraction)
    ConfigurableAxis axisRingCounts{"axisRingCounts", {90, -4.5, 4.5}, "<#it{R}>"};
    ConfigurableAxis axisDeltaCollisionIndex{"axisDeltaCollisionIndex", {2000, -0.5f, 1999.5f}, "#Delta collision index"}; // Always positive: SameKindPair pairs strictly upper. 2000 should cover the whole dataframe extension.
    ConfigurableAxis axisDeltaCollisionIndexNonAbs{"axisDeltaCollisionIndexNonAbs", {4000, -2000.5f, 1999.5f}, "#Delta collision index non abs"}; // Can be negative: the symmetric pairing calls of ReservoirInsert allow for preceding indices to be selected
    // AP plot axes:
    ConfigurableAxis axisAPAlpha{"axisAPAlpha", {220, -1.1f, 1.1f}, "V0 AP alpha"};
    ConfigurableAxis axisAPQt{"axisAPQt", {220, 0.0f, 0.5f}, "V0 AP alpha"};

    // Source-vs-target comparison for event mixing QA:
    ConfigurableAxis axisMixDeltaPt{"axisMixDeltaPt", {800, -40.f, 40.f}, "#Delta p_{T} (source - target) (GeV/c)"};
    ConfigurableAxis axisMixDeltaZvtx{"axisMixDeltaZvtx", {400, -20.f, 20.f}, "#Delta Z_{Vtx} (source - target) (cm)"};
    ConfigurableAxis axisMixDeltaCentrality{"axisMixDeltaCentrality", {1000, -100.f, 100.f}, "#Delta Centrality (source - target) (%)"};
    ConfigurableAxis axisMixDeltaEta{"axisMixDeltaEta", {90, -1.8f, 1.8f}, "#Delta#eta (source - target)"};
    ConfigurableAxis axisMixDeltaPhi{"axisMixDeltaPhi", {90, -constants::math::PI, constants::math::PI}, "#Delta#varphi (source - target)"};
    ConfigurableAxis axisMixCandidates{"axisMixCandidates", {200, -0.5f, 199.5f}, "Mixing candidates seen by this collision"};

    // Coarse axes for some index Vs "Delta mixed variable" TH2s:
    ConfigurableAxis axisMixDeltaIndexCoarse{"axisMixDeltaIndexCoarse", {401, -200.5f, 200.5f}, "#Delta collision index (target - source)"}; // Unit-wide bins; wider separations land in over/underflow
    ConfigurableAxis axisMixDeltaPtCoarse{"axisMixDeltaPtCoarse", {400, -40.f, 40.f}, "#Delta p_{T} (source - target) (GeV/c)"};
    ConfigurableAxis axisMixDeltaZvtxCoarse{"axisMixDeltaZvtxCoarse", {200, -1.f, 1.f}, "#Delta Z_{Vtx} (source - target) (cm)"};       // Support is one axisPVz bin, so +-0.5 cm
    ConfigurableAxis axisMixDeltaCentCoarse{"axisMixDeltaCentCoarse", {200, -100.f, 100.f}, "#Delta Centrality (source - target) (%)"};

    ConfigurableAxis axisDCAdau{"axisDCAdau", {10, 0., 2.0}, "DCA V0 daughters (cm)"};
    ConfigurableAxis axisDCAdauPV{"axisDCAdauPV", {10, 0., 1.2}, "DCA dauPV (cm)"}; // v0Selections.dcav0dau's default maximum is 1.2f in the TableProducer

    // Coarser axes for signal extraction:
    ConfigurableAxis axisPtSigExtract{"axisPtSigExtract", {VARIABLE_WIDTH, 0.0f, 0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 2.0f, 2.5f, 3.0f, 4.0f, 6.0f, 8.0f, 10.0f, 15.0f, 20.0f, 30.0f, 50.0f}, "pt axis for signal extraction"};
    // ConfigurableAxis axisLambdaMassSigExtract{"axisLambdaMassSigExtract", {175, 1.08f, 1.15f}, "Lambda mass in GeV/c"}; // With a sigma of 0.002 GeV/c, this has about 5 bins per sigma, so that the window is properly grasped.
    // A coarser axis:
    ConfigurableAxis axisLambdaMassSigExtract{
      "axisLambdaMassSigExtract",
      {VARIABLE_WIDTH,
      // Left sideband: 2 bins -- QA and sideband lever arm
      1.07800, 1.08682,
      // Fine region: 45 bins of 0.5 sigma, covering mu +/- 11.25 sigma (mu ~ 1.11537, sigma ~ 0.001745)
      1.09577, 1.09664, 1.09751, 1.09838, 1.09925, 1.10012,
      1.10099, 1.10186, 1.10273, 1.10360, 1.10447, 1.10534,
      1.10621, 1.10708, 1.10796, 1.10883, 1.10970, 1.11057,
      1.11145, 1.11232, 1.11319, 1.11406, 1.11494, 1.11581,
      1.11668, 1.11755, 1.11843, 1.11930, 1.12017, 1.12104,
      1.12192, 1.12279, 1.12366, 1.12453, 1.12541, 1.12628,
      1.12715, 1.12802, 1.12889, 1.12976, 1.13063, 1.13150,
      1.13237, 1.13324, 1.13411, 1.13498,
      // Right sideband: 2 bins
      1.14343, 1.15200},
      "Lambda mass in GeV/c"};
    // An axis with just the peak and the sidebands for signal-extraction-like QA with lots of statistics:
    // (edges are approximately +/- 3 sigma around the peak)
    ConfigurableAxis axisLambdaMassThreeBin{"axisLambdaMassThreeBin", {VARIABLE_WIDTH, 1.08f, 1.11014f, 1.12061f, 1.15f}, "m_{p#pi} (GeV/c^{2}), peak and sidebands"};
    // ConfigurableAxis axisLeadingParticlePtSigExtract{"axisLeadingParticlePtSigExtract", {VARIABLE_WIDTH, 0, 4, 8, 12, 16, 20, 25, 30, 35, 40, 60, 100, 200}, "Leading particle p_{T} (GeV/c)"}; // Simpler version!

    // (TODO: add a lambdaPt axis that is pre-selected only on the 0.5 to 1.5 Pt region for the Ring observable with lambda cuts to not store a huge histogram with empty bins by construction)

    // ConfigurableAxis axisCentrality{"axisCentrality", {VARIABLE_WIDTH, 0.0f, 5.0f, 10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f, 90.0f, 100.0f}, "Centrality"};
    ConfigurableAxis axisCentrality{"axisCentrality", {VARIABLE_WIDTH, 0.0f, 20.0f, 50.0f, 100.0f}, "Centrality"};

    // For the delta method error propagation (slightly better than just SEM error propagation with TProfiles):
    ConfigurableAxis axisDeltaComponents{"axisDeltaComponents", {5, 0.0, 5.0}, "0: r_k, 1: n_k, 2: r_k^2, 3: n_k^2, 4: r_k*n_k"};
  } axisConfigurations;

  // ===============================
  // Helper structs
  // ===============================
  // (Kept together, so everything the task carries around can be read in one place)

  // A small tracker struct for convenience -- Accumulates values for the Delta Method error estimator:
  struct EventDeltaTracker {
    double rInt = 0.0; // Ring accumulator
    int nInt = 0;      // Counts accumulator
    std::vector<double> rPt, rMass, rDtheta;
    std::vector<int> nPt, nMass, nDtheta;

    /// \brief Resizes every accumulator. Size includes ROOT's under/overflow bins, so the indices
    ///        returned by TAxis::FindBin() (0 .. nBins+1) can be used directly for dereferencing here.
    void resize(int nBinsPt, int nBinsMass, int nBinsDTheta)
    {
      rPt.assign(nBinsPt + 2, 0.0);
      rMass.assign(nBinsMass + 2, 0.0);
      rDtheta.assign(nBinsDTheta + 2, 0.0);
      nPt.assign(nBinsPt + 2, 0);
      nMass.assign(nBinsMass + 2, 0);
      nDtheta.assign(nBinsDTheta + 2, 0);
    }

    void reset()
    {
      rInt = 0.0;
      nInt = 0;
      std::fill(rPt.begin(), rPt.end(), 0.0);
      std::fill(rMass.begin(), rMass.end(), 0.0);
      std::fill(rDtheta.begin(), rDtheta.end(), 0.0);
      std::fill(nPt.begin(), nPt.end(), 0);
      std::fill(nMass.begin(), nMass.end(), 0);
      std::fill(nDtheta.begin(), nDtheta.end(), 0);
    }

    void addV0(double ringObs, int binPt, int binMass, int binDTheta)
    {
      rInt += ringObs;
      rPt[binPt] += ringObs;
      rMass[binMass] += ringObs;
      rDtheta[binDTheta] += ringObs;
      nInt += 1;
      nPt[binPt] += 1;
      nMass[binMass] += 1;
      nDtheta[binDTheta] += 1;
    }
  };

  /// \brief Kinematics of one jet proxy (leadP, leadJet or subJet). The same struct wherever a proxy is stored.
  /// \note  -999.f marks a value that was never filled (ALICE convention): e.g., pt = -999 for a leading particle
  ///        without a table entry in this collision.
  struct ProxyKinematics {
    bool isValid = false; //! passed the proxy's minimum pT. Re-evaluated by applyProxyDistortion
    float pt = -999.f;
    float eta = -999.f;
    float phi = -999.f;
  };

  /// \brief One proxy of each kind: the borrowed proxies of forcePreviousJet and doMixedEventProxies.
  struct SetOfProxies {
    ProxyKinematics leadP;
    ProxyKinematics leadJet;
    ProxyKinematics subJet;
  };

  /// \brief The collision-level mixing axes' cache. Cached once per collision so the pooling never needs a table iterator.
  struct MixingAxes {
    float zvtx = -999.f;
    float centrality = -999.f;
  };

  /// \brief One proxy's per-collision caches, shared by the main loop and the event mixing (one instance per proxy).
  /// \note  Direct-indexed by collision slot instead of hashed (at 1E6 collisions per dataframe, unordered_map did many
  ///        scattered allocations and lookup ended up with a costly DRAM access).
  struct ProxyByCollision {
    std::vector<ProxyKinematics> proxyKinematicsVec; //! the collision's own proxy, before any distortion
    std::vector<int32_t> mixedSourcesVec;           //! doMixedEventProxies LUT: nProxyResamples source slots per target (-1 = empty entry)
    std::vector<int32_t> candidateCountVec;         //! mixing candidates offered to each target (read back for QA)
  };

  // A simple struct for doMixedEventProxies. Stores the borrowed proxy and the event it came from.
  // (rebuilt on read from the source collision's caches -- the mixing LUTs only store the source slot to avoid cluttering memory with large nProxyResamples)
  struct MixedProxyInfo {
    ProxyKinematics proxy;     //! isValid stays false on an empty LUT entry
    float zvtx = -999.f;       //! Knowingly repeated for the LeadP, LeadJet and SubJet, so memory is occupied 3x for this same value.
    float centrality = -999.f;
    int64_t sourceCollisionId = -1;
  };

  /// \brief The per-V0 quantities the ring needs, computed once per V0 and shared by every proxy.
  struct V0RingInputs {
    XYZVector lambdaLike3Vec;
    XYZVector lambdaLikeUnit3Vec;
    XYZVector protonLikeStarUnit3Vec;
    float v0phi;
    float v0p; //! |p_Lambda|, only used for the R_z bearing chi
    float v0px;
    float v0py;
    float ringPrefactor;
    float protonStarRingBeamComponent;     //! p_D^* on the beam axis (R_z) or on the beam axis made transverse to the Lambda (R_z^T)
    float protonStarHelicityBeamComponent; //! the helicity part that R_z^T removes
  };

  /// \brief The ring observable and its companions, for one proxy and one V0. The defaults are the values of a missing proxy.
  struct ProxyRingCache {
    float ringObservable = 0.;
    float deltaPhi = 0.;
    float deltaTheta = 0.;
    float cosDeltaTheta = 0.;
    // KappaEff moments (cross moment u v and projection weight w):
    float kappaNum = 0.;
    float kappaDen = 1.;
    // Longitudinal-ring diagnostics (R_z and R_z^T only):
    float ringPerp = 0.;
    float helicityTerm = 0.;
    float chi = 0.;
    float nz = 0.;
  };

  // Allocating one tracker per family so the accumulators are allocated only once:
  EventDeltaTracker trackRing, trackRingKinCuts, trackJetKinCuts, trackJetLambdaKinCuts;

  // Axis pointers for Delta Method binning (fetched once in init, declared once here)
  TAxis* mAxisPt = nullptr;
  TAxis* mAxisMass = nullptr;
  TAxis* mAxisDTheta = nullptr;

  // V0 phi selection:
  std::vector<float> v0PhiLimitsVec;

  // Ring definition for the whole run: the full ring (default), R_z = P_z n_z, or R_z^T = R_z minus its helicity term.
  // Both longitudinal definitions share every code path but the beam-axis component of p* (set once in init):
  bool longitudinalRingMode = false;
  bool transverseBeamAxisMode = false;

  // Helper functions:
  // Fast wrapping into [-PI, PI) (restricted to this interval for function speed)
  inline double wrapToPiFast(double phi)
  {
    constexpr double TwoPi = constants::math::TwoPI;
    constexpr double Pi = constants::math::PI;
    if (phi >= Pi)
      phi -= TwoPi;
    else if (phi < -Pi)
      phi += TwoPi;
    return phi;
  }

  void init(InitContext const&)
  {
    // Configuration validation:
    // These combinations would not crash otherwise, so they are rejected at init() time.
    const int nDistortionsOn = static_cast<int>(fakePolSwitches.forcePerpToJet) + static_cast<int>(fakePolSwitches.forceJetDirectionSmudge) +
                               static_cast<int>(fakePolSwitches.forceRandJet) + static_cast<int>(fakePolSwitches.forcePreviousJet) +
                               static_cast<int>(fakePolSwitches.forceDatalikeJet) + static_cast<int>(fakePolSwitches.doMixedEventProxies);
    if (nDistortionsOn > 1) // applyProxyDistortion() is an if/else chain, so extra switches are silently ignored
      LOG(fatal) << "fakePolSwitches: " << nDistortionsOn << " proxy distortions enabled at once. They are mutually exclusive -- applyProxyDistortion() would apply only the first and silently drop the rest.";
    if (fakePolSwitches.nProxyResamples > 1 && fakePolSwitches.forcePreviousJet)
      LOG(fatal) << "fakePolSwitches: nProxyResamples > 1 is meaningless for forcePreviousJet. The previous jet does not change between resamplings, so every extra resampling pass would double-count the same proxy.";
    // A target sees at most mixedEventWindowSize partners ahead and as many behind, and sources are never repeated within a target:
    if (fakePolSwitches.doMixedEventProxies && fakePolSwitches.nProxyResamples > 2 * fakePolSwitches.mixedEventWindowSize)
      LOG(fatal) << "fakePolSwitches: nProxyResamples (" << fakePolSwitches.nProxyResamples << ") exceeds 2*mixedEventWindowSize (" << 2 * fakePolSwitches.mixedEventWindowSize << "). There are not as many available proxies in the window as requested, by construction.";
    if (excludeOutOfPeakQA && excludeInPeakQA) // Complementary selections (disjoint)
      LOG(fatal) << "excludeOutOfPeakQA and excludeInPeakQA are complementary: enabling both rejects every V0.";
    if (!analyseLambda && !analyseAntiLambda)
      LOG(fatal) << "analyseLambda and analyseAntiLambda are both false: no V0 would ever be analysed.";
    if (!familySwitches.doFamilyRing) // TODO: think of a smarter way of handling the axis getters for the DeltaMethod
      LOG(fatal) << "doFamilyRing must be on: the Delta Method accumulators take their binning from the Ring/ histograms.";
    if (std::abs((SidebandOuterNSigma - SidebandInnerNSigma) - PeakWindowNSigma) >= 1e-9)
      LOG(fatal) << "Sideband and peak windows must have the same width for the histograms here.";

    if (useRingZ && useRingZT) {
      LOG(fatal) << "useRingZ and useRingZT are mutually exclusive ring definitions: enable at most one.";
    }
    longitudinalRingMode = useRingZ || useRingZT;
    transverseBeamAxisMode = useRingZT;

    // V0 phi selection:
    v0PhiLimitsVec = (std::vector<float>)analysisLevelCuts.v0PhiLimits;

    // Ring observable histograms:
    // Helper to register one full histogram family (kinematic cut variation of ring observable)
    auto addRingObservableFamily = [&](const std::string& folder) {
      // ===============================
      // QA histograms: angle and pT distributions
      // (No mass dependency -- useful to check kinematic sculpting from cuts)
      // ===============================
      histosRingFamily.add((folder + "/LeadJet/QA/hDeltaPhi").c_str(), "#Delta#varphi_{jet};#Delta#varphi_{jet};Counts", kTH1D, {axisConfigurations.axisDeltaPhi});
      histosRingFamily.add((folder + "/LeadJet/QA/h2dDeltaPhiVsDeltaEta").c_str(), "#Delta#varphi_{jet};#Delta#varphi_{jet}; #eta_{#Lambda}-#eta_{Jet};Counts", kTH2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDeltaEtaCoarse});
      histosRingFamily.add((folder + "/LeadJet/QA/h2dDeltaPhiVsProxyPhi").c_str(), "#Delta#varphi_{jet};#varphi_{Jet};#varphi_{Jet};Counts", kTH2D, {axisConfigurations.axisPhi, axisConfigurations.axisDeltaPhi});
      histosRingFamily.add((folder + "/LeadJet/QA/hDeltaTheta").c_str(), "#Delta#theta_{jet};#Delta#theta_{jet};Counts", kTH1D, {axisConfigurations.axisDeltaTheta});
      histosRingFamily.add((folder + "/LeadJet/QA/hCosDeltaTheta").c_str(), "cos(#Delta#theta_{jet});cos(#Delta#theta_{jet});Counts", kTH1D, {axisConfigurations.axisCosTheta}); // Should actually be flat due to the geometry
      histosRingFamily.add((folder + "/LeadJet/QA/hIntegrated").c_str(), "Integrated counts; ;Counts", kTH1D, {{1, -0.5, 0.5}});

      histosRingFamily.add((folder + "/LeadP/QA/hDeltaPhi").c_str(), "#Delta#varphi_{LeadP};#Delta#varphi_{LeadP};Counts", kTH1D, {axisConfigurations.axisDeltaPhi});
      histosRingFamily.add((folder + "/LeadP/QA/hDeltaTheta").c_str(), "#Delta#theta_{LeadP};#Delta#theta_{LeadP};Counts", kTH1D, {axisConfigurations.axisDeltaTheta});
      histosRingFamily.add((folder + "/LeadP/QA/hCosDeltaTheta").c_str(), "cos(#Delta#theta_{LeadP});cos(#Delta#theta_{LeadP});Counts", kTH1D, {axisConfigurations.axisCosTheta}); // Should actually be flat due to the geometry
      histosRingFamily.add((folder + "/SubJet/QA/hDeltaPhi").c_str(), "#Delta#varphi_{SubJet};#Delta#varphi_{SubJet};Counts", kTH1D, {axisConfigurations.axisDeltaPhi});
      histosRingFamily.add((folder + "/SubJet/QA/hDeltaTheta").c_str(), "#Delta#theta_{SubJet};#Delta#theta_{SubJet};Counts", kTH1D, {axisConfigurations.axisDeltaTheta});
      histosRingFamily.add((folder + "/SubJet/QA/hCosDeltaTheta").c_str(), "cos(#Delta#theta_{SubJet});cos(#Delta#theta_{SubJet});Counts", kTH1D, {axisConfigurations.axisCosTheta}); // Should actually be flat due to the geometry

      // ===============================
      // Lambda pT dependence
      // ===============================
      histosRingFamily.add((folder + "/LeadJet/QA/hLambdaPt").c_str(), "#Lambda #it{p}_{T};#it{p}_{T}^{#Lambda} (GeV/c);Counts", kTH1D, {axisConfigurations.axisPt});
      histosRingFamily.add((folder + "/LeadJet/QA/h2dDeltaPhiVsLambdaPt").c_str(), "#Delta#varphi_{jet} vs #Lambda #it{p}_{T};#Delta#varphi_{jet};#it{p}_{T}^{#Lambda} (GeV/c)", kTH2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisPt});
      histosRingFamily.add((folder + "/LeadJet/QA/h2dDeltaThetaVsLambdaPt").c_str(), "#Delta#theta_{jet} vs #Lambda #it{p}_{T};#Delta#theta_{jet};#it{p}_{T}^{#Lambda} (GeV/c)", kTH2D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisPt});
      // ===============================
      //   Polarization observable QAs
      // (not Ring: actual polarization!)
      // ===============================
      // Will implement these as TProfiles, as polarization is also a measure like P_Lambda = (3/\alpha_Lambda) * <p_{proton}>, so the error is similar
      // ===============================
      // 1D TProfiles
      // ===============================
      histosRingFamily.add((folder + "/QA/pPxStarPhi").c_str(), "<P_{#Lambda}^{*}>_{x} vs #varphi_{#Lambda};#varphi_{#Lambda};<P_{#Lambda}^{*}>_{x}", kTProfile, {axisConfigurations.axisDeltaPhi});
      histosRingFamily.add((folder + "/QA/pPyStarPhi").c_str(), "<P_{#Lambda}^{*}>_{y} vs #varphi_{#Lambda};#varphi_{#Lambda};<P_{#Lambda}^{*}>_{y}", kTProfile, {axisConfigurations.axisDeltaPhi});
      histosRingFamily.add((folder + "/QA/pPzStarPhi").c_str(), "<P_{#Lambda}^{*}>_{z} vs #varphi_{#Lambda};#varphi_{#Lambda};<P_{#Lambda}^{*}>_{z}", kTProfile, {axisConfigurations.axisDeltaPhi});
      histosRingFamily.add((folder + "/LeadJet/QA/pPxStarDeltaPhi").c_str(), "<P_{#Lambda}^{*}>_{x} vs #Delta#varphi_{jet};#Delta#varphi_{jet};<P_{#Lambda}^{*}>_{x}", kTProfile, {axisConfigurations.axisDeltaPhi});
      histosRingFamily.add((folder + "/LeadJet/QA/pPyStarDeltaPhi").c_str(), "<P_{#Lambda}^{*}>_{y} vs #Delta#varphi_{jet};#Delta#varphi_{jet};<P_{#Lambda}^{*}>_{y}", kTProfile, {axisConfigurations.axisDeltaPhi});
      histosRingFamily.add((folder + "/LeadJet/QA/pPzStarDeltaPhi").c_str(), "<P_{#Lambda}^{*}>_{z} vs #Delta#varphi_{jet};#Delta#varphi_{jet};<P_{#Lambda}^{*}>_{z}", kTProfile, {axisConfigurations.axisDeltaPhi});
      // Profiles of polarization Vs AEE angle:
      // (there should be NO dependence on the AEE angle for the Z-axis polarization)
      histosRingFamily.add((folder + "/QA/pPxStarPhiAEE").c_str(), "<P_{#Lambda}^{*}>_{x} vs AEE angle;#phi_{#Lambda}-#phi_{p}^{*};<P_{#Lambda}^{*}>_{x}", kTProfile, {axisConfigurations.axisDeltaPhi});
      histosRingFamily.add((folder + "/QA/pPyStarPhiAEE").c_str(), "<P_{#Lambda}^{*}>_{y} vs AEE angle;#phi_{#Lambda}-#phi_{p}^{*};<P_{#Lambda}^{*}>_{y}", kTProfile, {axisConfigurations.axisDeltaPhi});
      histosRingFamily.add((folder + "/QA/pPzStarPhiAEE").c_str(), "<P_{#Lambda}^{*}>_{z} vs AEE angle;#phi_{#Lambda}-#phi_{p}^{*};<P_{#Lambda}^{*}>_{z}", kTProfile, {axisConfigurations.axisDeltaPhi});
      // ===============================
      // 2D TProfiles (Lambda correlations)
      // ===============================
      histosRingFamily.add((folder + "/LeadJet/QA/p2dPxStarDeltaPhiVsLambdaPt").c_str(), "<P_{#Lambda}^{*}>_{x} vs #Delta#varphi_{jet} vs #it{p}_{T}^{#Lambda};#Delta#varphi_{jet};#it{p}_{T}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{x}", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisPtSigExtract});
      histosRingFamily.add((folder + "/LeadJet/QA/p2dPyStarDeltaPhiVsLambdaPt").c_str(), "<P_{#Lambda}^{*}>_{y} vs #Delta#varphi_{jet} vs #it{p}_{T}^{#Lambda};#Delta#varphi_{jet};#it{p}_{T}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{y}", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisPtSigExtract});
      histosRingFamily.add((folder + "/LeadJet/QA/p2dPzStarDeltaPhiVsLambdaPt").c_str(), "<P_{#Lambda}^{*}>_{z} vs #Delta#varphi_{jet} vs #it{p}_{T}^{#Lambda};#Delta#varphi_{jet};#it{p}_{T}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{z}", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisPtSigExtract});

      // =========================================================================================
      // 2D vector-field profiles (+QA TH2D counters) of the polarization, in four coordinate systems:
      // (a companion code plots these as a vector field)
      // Frames are:
      // - Lab - the detector frame's PxPyPz
      // - Aee - XY rotated about \hat z so that \hat x_Aee = \hat p_{proton, Transverse}^*
      // - PrimeV0 - XY rotated about \hat z so that \hat x_V0 = \hat p_{Lambda, Transverse}
      // - PrimeJet - \hat z_PrimeJet = \hat JetDirection and \hat x_PrimeJet is built based on the lab's \hat z direction
      // =========================================================================================
      // Lab frame:
      histosRingFamily.add((folder + "/PolMaps/Lab/p2dPxStar_vsPxPy").c_str(), "<P_{#Lambda}^{*}>_{x} vs (p_{x}^{#Lambda},p_{y}^{#Lambda});p_{x}^{#Lambda} (GeV/c);p_{y}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{x}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/PolMaps/Lab/p2dPyStar_vsPxPy").c_str(), "<P_{#Lambda}^{*}>_{y} vs (p_{x}^{#Lambda},p_{y}^{#Lambda});p_{x}^{#Lambda} (GeV/c);p_{y}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{y}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/PolMaps/Lab/p2dPzStar_vsPxPy").c_str(), "<P_{#Lambda}^{*}>_{z} vs (p_{x}^{#Lambda},p_{y}^{#Lambda});p_{x}^{#Lambda} (GeV/c);p_{y}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{z}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/PolMaps/Lab/h2dCountsVsPxPy").c_str(), "V0 Counts vs (p_{x}^{#Lambda},p_{y}^{#Lambda});p_{x}^{#Lambda} (GeV/c);p_{y}^{#Lambda} (GeV/c);Counts", kTH2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});

      histosRingFamily.add((folder + "/PolMaps/Lab/p2dPxStar_vsPzPx").c_str(), "<P_{#Lambda}^{*}>_{x} vs (p_{z}^{#Lambda},p_{x}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{x}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{x}", kTProfile2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/PolMaps/Lab/p2dPyStar_vsPzPx").c_str(), "<P_{#Lambda}^{*}>_{y} vs (p_{z}^{#Lambda},p_{x}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{x}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{y}", kTProfile2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/PolMaps/Lab/p2dPzStar_vsPzPx").c_str(), "<P_{#Lambda}^{*}>_{z} vs (p_{z}^{#Lambda},p_{x}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{x}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{z}", kTProfile2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/PolMaps/Lab/h2dCountsVsPzPx").c_str(), "V0 Counts vs (p_{z}^{#Lambda},p_{x}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{x}^{#Lambda} (GeV/c);Counts", kTH2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPRot});

      histosRingFamily.add((folder + "/PolMaps/Lab/p2dPxStar_vsPyPz").c_str(), "<P_{#Lambda}^{*}>_{x} vs (p_{y}^{#Lambda},p_{z}^{#Lambda});p_{y}^{#Lambda} (GeV/c);p_{z}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{x}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPz});
      histosRingFamily.add((folder + "/PolMaps/Lab/p2dPyStar_vsPyPz").c_str(), "<P_{#Lambda}^{*}>_{y} vs (p_{y}^{#Lambda},p_{z}^{#Lambda});p_{y}^{#Lambda} (GeV/c);p_{z}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{y}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPz});
      histosRingFamily.add((folder + "/PolMaps/Lab/p2dPzStar_vsPyPz").c_str(), "<P_{#Lambda}^{*}>_{z} vs (p_{y}^{#Lambda},p_{z}^{#Lambda});p_{y}^{#Lambda} (GeV/c);p_{z}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{z}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPz});
      histosRingFamily.add((folder + "/PolMaps/Lab/h2dCountsVsPyPz").c_str(), "V0 Counts vs (p_{y}^{#Lambda},p_{z}^{#Lambda});p_{y}^{#Lambda} (GeV/c);p_{z}^{#Lambda} (GeV/c);Counts", kTH2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPz});

      // Aee frame (coordinates where PhiAEE = 0 in direction \hat x'):
      // In this frame the transverse polarization is (|P*_T|, 0) by construction, so <P*_x> and <P*_y> are redundant and replaced by a transverse polarization:
      histosRingFamily.add((folder + "/PolMaps/Aee/p2dPtStar_vsPxAeePyAee").c_str(), "<P_{#Lambda}^{*}>_{T} vs (p_{x,AEE}^{#Lambda},p_{y,AEE}^{#Lambda});p_{x,AEE}^{#Lambda} (GeV/c);p_{y,AEE}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{T}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/PolMaps/Aee/p2dPzStar_vsPxAeePyAee").c_str(), "<P_{#Lambda}^{*}>_{z} vs (p_{x,AEE}^{#Lambda},p_{y,AEE}^{#Lambda});p_{x,AEE}^{#Lambda} (GeV/c);p_{y,AEE}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{z}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/PolMaps/Aee/h2dCountsVsPxAeePyAee").c_str(), "V0 Counts vs (p_{x,AEE}^{#Lambda},p_{y,AEE}^{#Lambda});p_{x,AEE}^{#Lambda} (GeV/c);p_{y,AEE}^{#Lambda} (GeV/c);Counts", kTH2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});

      histosRingFamily.add((folder + "/PolMaps/Aee/p2dPtStar_vsPzPxAee").c_str(), "<P_{#Lambda}^{*}>_{T} vs (p_{z}^{#Lambda},p_{x,AEE}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{x,AEE}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{T}", kTProfile2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/PolMaps/Aee/p2dPzStar_vsPzPxAee").c_str(), "<P_{#Lambda}^{*}>_{z} vs (p_{z}^{#Lambda},p_{x,AEE}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{x,AEE}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{z}", kTProfile2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/PolMaps/Aee/h2dCountsVsPzPxAee").c_str(), "V0 Counts vs (p_{z}^{#Lambda},p_{x,AEE}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{x,AEE}^{#Lambda} (GeV/c);Counts", kTH2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPRot});

      histosRingFamily.add((folder + "/PolMaps/Aee/p2dPtStar_vsPyAeePz").c_str(), "<P_{#Lambda}^{*}>_{T} vs (p_{y,AEE}^{#Lambda},p_{z}^{#Lambda});p_{y,AEE}^{#Lambda} (GeV/c);p_{z}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{T}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPz});
      histosRingFamily.add((folder + "/PolMaps/Aee/p2dPzStar_vsPyAeePz").c_str(), "<P_{#Lambda}^{*}>_{z} vs (p_{y,AEE}^{#Lambda},p_{z}^{#Lambda});p_{y,AEE}^{#Lambda} (GeV/c);p_{z}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{z}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPz});
      histosRingFamily.add((folder + "/PolMaps/Aee/h2dCountsVsPyAeePz").c_str(), "V0 Counts vs (p_{y,AEE}^{#Lambda},p_{z}^{#Lambda});p_{y,AEE}^{#Lambda} (GeV/c);p_{z}^{#Lambda} (GeV/c);Counts", kTH2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPz});

      // PrimeV0 frame (\hat x'_V0 along the Lambda's transverse direction):
      // (<P*_y'V0> is the ring observable computed with the beam as the jet proxy)
      histosRingFamily.add((folder + "/PolMaps/PrimeV0/p2dPxStarPrimeV0_vsPzPt").c_str(), "<P_{#Lambda}^{*}>_{x'V0} vs (p_{z}^{#Lambda},p_{T}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{T}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{x'V0}", kTProfile2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPtRot});
      histosRingFamily.add((folder + "/PolMaps/PrimeV0/p2dPyStarPrimeV0_vsPzPt").c_str(), "<P_{#Lambda}^{*}>_{y'V0} vs (p_{z}^{#Lambda},p_{T}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{T}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{y'V0}", kTProfile2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPtRot});
      histosRingFamily.add((folder + "/PolMaps/PrimeV0/p2dPzStar_vsPzPt").c_str(), "<P_{#Lambda}^{*}>_{z} vs (p_{z}^{#Lambda},p_{T}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{T}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{z}", kTProfile2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPtRot});
      histosRingFamily.add((folder + "/PolMaps/PrimeV0/h2dCountsVsPzPt").c_str(), "V0 Counts vs (p_{z}^{#Lambda},p_{T}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{T}^{#Lambda} (GeV/c);Counts", kTH2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPtRot});

      // PrimeJet frame (\hat z_Jet = jet, \hat x_Jet = beam direction, orthogonalised against the jet)
      histosRingFamily.add((folder + "/PolMaps/PrimeJet/p2dPxStarPrimeJet_vsPxPyPrimeJet").c_str(), "<P_{#Lambda}^{*}>_{x'Jet} vs (p_{x'Jet}^{#Lambda},p_{y'Jet}^{#Lambda});p_{x'Jet}^{#Lambda} (GeV/c);p_{y'Jet}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{x'Jet}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/PolMaps/PrimeJet/p2dPyStarPrimeJet_vsPxPyPrimeJet").c_str(), "<P_{#Lambda}^{*}>_{y'Jet} vs (p_{x'Jet}^{#Lambda},p_{y'Jet}^{#Lambda});p_{x'Jet}^{#Lambda} (GeV/c);p_{y'Jet}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{y'Jet}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/PolMaps/PrimeJet/p2dPzStarPrimeJet_vsPxPyPrimeJet").c_str(), "<P_{#Lambda}^{*}>_{z'Jet} vs (p_{x'Jet}^{#Lambda},p_{y'Jet}^{#Lambda});p_{x'Jet}^{#Lambda} (GeV/c);p_{y'Jet}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{z'Jet}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/PolMaps/PrimeJet/h2dCountsVsPxPyPrimeJet").c_str(), "V0 Counts vs (p_{x'Jet}^{#Lambda},p_{y'Jet}^{#Lambda});p_{x'Jet}^{#Lambda} (GeV/c);p_{y'Jet}^{#Lambda} (GeV/c);Counts", kTH2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});

      histosRingFamily.add((folder + "/PolMaps/PrimeJet/p2dPxStarPrimeJet_vsPzPxPrimeJet").c_str(), "<P_{#Lambda}^{*}>_{x'Jet} vs (p_{z'Jet}^{#Lambda},p_{x'Jet}^{#Lambda});p_{z'Jet}^{#Lambda} (GeV/c);p_{x'Jet}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{x'Jet}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/PolMaps/PrimeJet/p2dPyStarPrimeJet_vsPzPxPrimeJet").c_str(), "<P_{#Lambda}^{*}>_{y'Jet} vs (p_{z'Jet}^{#Lambda},p_{x'Jet}^{#Lambda});p_{z'Jet}^{#Lambda} (GeV/c);p_{x'Jet}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{y'Jet}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/PolMaps/PrimeJet/p2dPzStarPrimeJet_vsPzPxPrimeJet").c_str(), "<P_{#Lambda}^{*}>_{z'Jet} vs (p_{z'Jet}^{#Lambda},p_{x'Jet}^{#Lambda});p_{z'Jet}^{#Lambda} (GeV/c);p_{x'Jet}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{z'Jet}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/PolMaps/PrimeJet/h2dCountsVsPzPxPrimeJet").c_str(), "V0 Counts vs (p_{z'Jet}^{#Lambda},p_{x'Jet}^{#Lambda});p_{z'Jet}^{#Lambda} (GeV/c);p_{x'Jet}^{#Lambda} (GeV/c);Counts", kTH2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});

      histosRingFamily.add((folder + "/PolMaps/PrimeJet/p2dPxStarPrimeJet_vsPyPzPrimeJet").c_str(), "<P_{#Lambda}^{*}>_{x'Jet} vs (p_{y'Jet}^{#Lambda},p_{z'Jet}^{#Lambda});p_{y'Jet}^{#Lambda} (GeV/c);p_{z'Jet}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{x'Jet}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/PolMaps/PrimeJet/p2dPyStarPrimeJet_vsPyPzPrimeJet").c_str(), "<P_{#Lambda}^{*}>_{y'Jet} vs (p_{y'Jet}^{#Lambda},p_{z'Jet}^{#Lambda});p_{y'Jet}^{#Lambda} (GeV/c);p_{z'Jet}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{y'Jet}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/PolMaps/PrimeJet/p2dPzStarPrimeJet_vsPyPzPrimeJet").c_str(), "<P_{#Lambda}^{*}>_{z'Jet} vs (p_{y'Jet}^{#Lambda},p_{z'Jet}^{#Lambda});p_{y'Jet}^{#Lambda} (GeV/c);p_{z'Jet}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{z'Jet}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/PolMaps/PrimeJet/h2dCountsVsPyPzPrimeJet").c_str(), "V0 Counts vs (p_{y'Jet}^{#Lambda},p_{z'Jet}^{#Lambda});p_{y'Jet}^{#Lambda} (GeV/c);p_{z'Jet}^{#Lambda} (GeV/c);Counts", kTH2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});

      // ===============================
      // Ring observable, single scalar 2D profile:
      // (no need to rotate before calculating <R>, as it is invariant by rotation)
      // ===============================
      histosRingFamily.add((folder + "/LeadJet/RingMaps/p2dRingObservableVsPxPy").c_str(), "<#it{R}> vs (p_{x}^{#Lambda},p_{y}^{#Lambda});p_{x}^{#Lambda} (GeV/c);p_{y}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/LeadJet/RingMaps/p2dRingObservableVsPzPx").c_str(), "<#it{R}> vs (p_{z}^{#Lambda},p_{x}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{x}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/LeadJet/RingMaps/p2dRingObservableVsPyPz").c_str(), "<#it{R}> vs (p_{y}^{#Lambda},p_{z}^{#Lambda});p_{y}^{#Lambda} (GeV/c);p_{z}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPz});
      // On the AEE planes:
      histosRingFamily.add((folder + "/LeadJet/RingMaps/p2dRingObservableVsPxAeePyAee").c_str(), "<#it{R}> vs (p_{x,AEE}^{#Lambda},p_{y,AEE}^{#Lambda});p_{x,AEE}^{#Lambda} (GeV/c);p_{y,AEE}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/LeadJet/RingMaps/p2dRingObservableVsPzPxAee").c_str(), "<#it{R}> vs (p_{z}^{#Lambda},p_{x,AEE}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{x,AEE}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/LeadJet/RingMaps/p2dRingObservableVsPyAeePz").c_str(), "<#it{R}> vs (p_{y,AEE}^{#Lambda},p_{z}^{#Lambda});p_{y,AEE}^{#Lambda} (GeV/c);p_{z}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPz});
      // On the jet-frame planes:
      histosRingFamily.add((folder + "/LeadJet/RingMaps/p2dRingObservableVsPxPyPrimeJet").c_str(), "<#it{R}> vs (p_{x'Jet}^{#Lambda},p_{y'Jet}^{#Lambda});p_{x'Jet}^{#Lambda} (GeV/c);p_{y'Jet}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/LeadJet/RingMaps/p2dRingObservableVsPzPxPrimeJet").c_str(), "<#it{R}> vs (p_{z'Jet}^{#Lambda},p_{x'Jet}^{#Lambda});p_{z'Jet}^{#Lambda} (GeV/c);p_{x'Jet}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/LeadJet/RingMaps/p2dRingObservableVsPyPzPrimeJet").c_str(), "<#it{R}> vs (p_{y'Jet}^{#Lambda},p_{z'Jet}^{#Lambda});p_{y'Jet}^{#Lambda} (GeV/c);p_{z'Jet}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      // For LeadP estimators:
      histosRingFamily.add((folder + "/LeadP/RingMaps/p2dRingObservableVsPxPy").c_str(), "<#it{R}>_{LeadP} vs (p_{x}^{#Lambda},p_{y}^{#Lambda});p_{x}^{#Lambda} (GeV/c);p_{y}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/LeadP/RingMaps/p2dRingObservableVsPzPx").c_str(), "<#it{R}>_{LeadP} vs (p_{z}^{#Lambda},p_{x}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{x}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/LeadP/RingMaps/p2dRingObservableVsPyPz").c_str(), "<#it{R}>_{LeadP} vs (p_{y}^{#Lambda},p_{z}^{#Lambda});p_{y}^{#Lambda} (GeV/c);p_{z}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPz});

      // TProfiles with correct error bars::
      // -- TProfiles will handle the error estimate of the Ring Observable via the variance, even though
      // they still lack the proper signal extraction and possible efficiency corrections in the current state
      // -- If any efficiency corrections arise, you can fill with the kTH1D as (deltaPhiJet, ringObservable, weight)
      // instead of the simple (deltaPhiJet, ringObservable) --> Notice TProfile knows how to accept 3 entries
      // for a TH1D-like object!
      // -- CAUTION! The TProfile does not utilize unbiased variance estimators with N-1 instead of N in the denominator,
      // so you might get biased errors when counts are too low in higher-dimensional profiles (i.e., kTProfile2Ds)
      // ===============================
      // 1D TProfiles
      // ===============================
      histosRingFamily.add((folder + "/LeadJet/pRingObservableDeltaPhi").c_str(), "<#it{R}> vs #Delta#varphi_{jet};#Delta#varphi_{jet};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      // To see the actual distribution of counts in data (another differential-like shape of the distribution we are taking an average of):
      histosRingFamily.add((folder + "/LeadJet/hRingObservableCounts").c_str(), "Counts vs <#it{R}>_{jet};<#it{R}>; Counts", kTH1D, {axisConfigurations.axisRingCounts});
      histosRingFamily.add((folder + "/LeadJet/pRingObservableProxyPhi").c_str(), "<#it{R}> vs #varphi_{jet};#varphi_{jet};<#it{R}>", kTProfile, {axisConfigurations.axisPhi});
      histosRingFamily.add((folder + "/LeadJet/pRingObservablePhiLambda").c_str(), "<#it{R}> vs #varphi_{#Lambda};#varphi_{#Lambda};<#it{R}>", kTProfile, {axisConfigurations.axisPhi});
      // ===============================
      // Ring projection kernel
      // The two variables <R> depends on analytically, plus the Lambda direction cosine in the 3D version,
      // which is what lets the eta_Lambda sample be symmetrised downstream instead of at fill time.
      // ===============================
      // Binned in the two variables the projection is analytically linear in -- cos(DeltaTheta) and the
      // proxy direction cosine t_z -- rather than in DeltaTheta and eta. Any fitting of the resulting
      // surface belongs downstream, in the post-processing macros.
      histosRingFamily.add((folder + "/LeadJet/RingKernel/p2dRingObservableCosDeltaThetaVsProxyZ").c_str(), "<#it{R}> vs (cos#Delta#theta_{jet},#hat{t}_{z});cos#Delta#theta_{jet};#hat{t}_{z};<#it{R}>", kTProfile2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisJetZ});
      histosRingFamily.add((folder + "/LeadJet/RingKernel/h2dCountsCosDeltaThetaVsProxyZ").c_str(), "Counts vs (cos#Delta#theta_{jet},#hat{t}_{z});cos#Delta#theta_{jet};#hat{t}_{z};Counts", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisJetZ});
      histosRingFamily.add((folder + "/LeadJet/RingKernel/p3dRingObservableCosDeltaThetaVsProxyZVsLambdaZ").c_str(), "<#it{R}> vs (cos#Delta#theta_{jet},#hat{t}_{z},cos#theta_{#Lambda});cos#Delta#theta_{jet};#hat{t}_{z};cos#theta_{#Lambda}", kTProfile3D, {axisConfigurations.axisCosThetaCoarse, axisConfigurations.axisJetZ, axisConfigurations.axisLambdaZ});

      // LeadP proxy. B_phi knows nothing about the reference axis, so this must return the same kernel as the jet version -- with a wider t_z lever.
      histosRingFamily.add((folder + "/LeadP/RingKernel/p2dRingObservableCosDeltaThetaVsProxyZ").c_str(), "<#it{R}>_{LeadP} vs (cos#Delta#theta_{LeadP},#hat{t}_{z}^{LeadP});cos#Delta#theta_{LeadP};#hat{t}_{z}^{LeadP};<#it{R}>", kTProfile2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisLeadPZ});
      histosRingFamily.add((folder + "/LeadP/RingKernel/h2dCountsCosDeltaThetaVsProxyZ").c_str(), "Counts vs (cos#Delta#theta_{LeadP},#hat{t}_{z}^{LeadP});cos#Delta#theta_{LeadP};#hat{t}_{z}^{LeadP};Counts", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisLeadPZ});
      histosRingFamily.add((folder + "/LeadP/RingKernel/p3dRingObservableCosDeltaThetaVsProxyZVsLambdaZ").c_str(), "<#it{R}>_{LeadP} vs (cos#Delta#theta_{LeadP},#hat{t}_{z}^{LeadP},cos#theta_{#Lambda});cos#Delta#theta_{LeadP};#hat{t}_{z}^{LeadP};cos#theta_{#Lambda}", kTProfile3D, {axisConfigurations.axisCosThetaCoarse, axisConfigurations.axisLeadPZ, axisConfigurations.axisLambdaZ});

      histosRingFamily.add((folder + "/LeadJet/pRingObservableDeltaTheta").c_str(), "<#it{R}> vs #Delta#theta_{jet};#Delta#theta_{jet};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaTheta});
      histosRingFamily.add((folder + "/LeadJet/pRingObservableIntegrated").c_str(), "Integrated <#it{R}>; ;<#it{R}>", kTProfile, {{1, -0.5, 0.5}});
      histosRingFamily.add((folder + "/LeadJet/pRingObservableLambdaPt").c_str(), "<#it{R}> vs #it{p}_{T}^{#Lambda};#it{p}_{T}^{#Lambda} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisPt});

      // Ring vs Jet proxy pT:
      histosRingFamily.add((folder + "/LeadJet/ProxyPtDependence/pRingVsProxyPt").c_str(), "<#it{R}> vs Jet #it{p}_{T};#it{p}_{T}^{Jet} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
      histosRingFamily.add((folder + "/LeadP/ProxyPtDependence/pRingVsProxyPt").c_str(), "<#it{R}> vs LeadP #it{p}_{T};#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
      histosRingFamily.add((folder + "/SubJet/ProxyPtDependence/pRingVsProxyPt").c_str(), "<#it{R}> vs SubJet #it{p}_{T};#it{p}_{T}^{SubJet} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
      // And some counters to be aware of the amount of Lambdas (and jets) in each pT interval:
      histosRingFamily.add((folder + "/LeadJet/QA/hProxyPt").c_str(), "Jet #it{p}_{T};#it{p}_{T}^{Jet} (GeV/c);Counts", kTH1D, {axisConfigurations.axisJetPt});
      histosRingFamily.add((folder + "/LeadP/QA/hProxyPt").c_str(), "LeadP #it{p}_{T};#it{p}_{T}^{LeadP} (GeV/c);Counts", kTH1D, {axisConfigurations.axisJetPt});
      histosRingFamily.add((folder + "/SubJet/QA/hProxyPt").c_str(), "SubJet #it{p}_{T};#it{p}_{T}^{SubJet} (GeV/c);Counts", kTH1D, {axisConfigurations.axisJetPt});

      // Splitting into positive and negative eta contributions:
      histosRingFamily.add((folder + "/LeadJet/ProxyPtDependence/p2dRingVsProxyPtVsProxyEta").c_str(), "<#it{R}> vs Jet #it{p}_{T} vs #eta_{Jet};#it{p}_{T}^{Jet} (GeV/c);#eta_{Jet};<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, {2, -0.9, 0.9}});
      histosRingFamily.add((folder + "/LeadP/ProxyPtDependence/p2dRingVsProxyPtVsProxyEta").c_str(), "<#it{R}> vs LeadP #it{p}_{T} vs #eta_{LeadP};#it{p}_{T}^{LeadP} (GeV/c);#eta_{LeadP};<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, {2, -0.9, 0.9}});
      histosRingFamily.add((folder + "/SubJet/ProxyPtDependence/p2dRingVsProxyPtVsProxyEta").c_str(), "<#it{R}> vs SubJet #it{p}_{T} vs #eta_{SubJet};#it{p}_{T}^{SubJet} (GeV/c);#eta_{SubJet};<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, {2, -0.9, 0.9}});
      // For each Lambda's eta:
      histosRingFamily.add((folder + "/LeadJet/ProxyPtDependence/p2dRingVsProxyPtVsEtaV0").c_str(), "<#it{R}> vs Jet #it{p}_{T} vs #eta_{V0};#it{p}_{T}^{Jet} (GeV/c);#eta_{V0};<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, {2, -0.9, 0.9}});
      histosRingFamily.add((folder + "/LeadP/ProxyPtDependence/p2dRingVsProxyPtVsEtaV0").c_str(), "<#it{R}> vs LeadP #it{p}_{T} vs #eta_{V0};#it{p}_{T}^{LeadP} (GeV/c);#eta_{V0};<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, {2, -0.9, 0.9}});
      histosRingFamily.add((folder + "/SubJet/ProxyPtDependence/p2dRingVsProxyPtVsEtaV0").c_str(), "<#it{R}> vs SubJet #it{p}_{T} vs #eta_{V0};#it{p}_{T}^{SubJet} (GeV/c);#eta_{V0};<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, {2, -0.9, 0.9}});

      // Rasterizing, only for LeadP the TProfile2D into two TProfile 1Ds (easier to draw with "same")
      histosRingFamily.add((folder + "/LeadP/ProxyPtDependence/pRingVsProxyPt_PosProxyEta").c_str(), "<#it{R}> vs LeadP #it{p}_{T} (#eta_{LeadP}>0);#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
      histosRingFamily.add((folder + "/LeadP/ProxyPtDependence/pRingVsProxyPt_NegProxyEta").c_str(), "<#it{R}> vs LeadP #it{p}_{T} (#eta_{LeadP}<0);#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
      // V0 eta:
      histosRingFamily.add((folder + "/LeadP/ProxyPtDependence/pRingVsProxyPt_PosEtaV0").c_str(), "<#it{R}> vs LeadP #it{p}_{T} (#eta_{V0}>0);#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
      histosRingFamily.add((folder + "/LeadP/ProxyPtDependence/pRingVsProxyPt_NegEtaV0").c_str(), "<#it{R}> vs LeadP #it{p}_{T} (#eta_{V0}<0);#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});

      // Looking at V0Eta and JetEta combinations (only for LeadP):
      histosRingFamily.add((folder + "/LeadP/ProxyPtDependence/pRingVsProxyPt_PosProxyEta_PosEtaV0").c_str(), "<#it{R}> vs LeadP #it{p}_{T} (#eta_{LeadP}>0, #eta_{V0}>0);#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
      histosRingFamily.add((folder + "/LeadP/ProxyPtDependence/pRingVsProxyPt_NegProxyEta_PosEtaV0").c_str(), "<#it{R}> vs LeadP #it{p}_{T} (#eta_{LeadP}<0, #eta_{V0}>0);#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
      histosRingFamily.add((folder + "/LeadP/ProxyPtDependence/pRingVsProxyPt_PosProxyEta_NegEtaV0").c_str(), "<#it{R}> vs LeadP #it{p}_{T} (#eta_{LeadP}>0, #eta_{V0}<0);#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
      histosRingFamily.add((folder + "/LeadP/ProxyPtDependence/pRingVsProxyPt_NegProxyEta_NegEtaV0").c_str(), "<#it{R}> vs LeadP #it{p}_{T} (#eta_{LeadP}<0, #eta_{V0}<0);#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});

      // Integrated <R> Vs ProxyPt Vs Centrality:
      histosRingFamily.add((folder + "/LeadJet/ProxyPtDependence/p2dRingVsProxyPtVsCentrality").c_str(), "<#it{R}> vs Jet #it{p}_{T} vs Centrality;#it{p}_{T}^{Jet} (GeV/c);Centrality(%);<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, axisConfigurations.axisCentrality});
      histosRingFamily.add((folder + "/LeadP/ProxyPtDependence/p2dRingVsProxyPtVsCentrality").c_str(), "<#it{R}> vs LeadP #it{p}_{T} vs Centrality;#it{p}_{T}^{LeadP} (GeV/c);Centrality(%);<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, axisConfigurations.axisCentrality});
      histosRingFamily.add((folder + "/SubJet/ProxyPtDependence/p2dRingVsProxyPtVsCentrality").c_str(), "<#it{R}> vs SubJet #it{p}_{T} vs Centrality;#it{p}_{T}^{SubJet} (GeV/c);Centrality(%);<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, axisConfigurations.axisCentrality});

      // Understanding eta dependence seen in pRingEtaCuts:
      histosRingFamily.add((folder + "/LeadJet/EtaDependence/pRingObservableEtaLambda").c_str(), "<#it{R}> vs #eta_{#Lambda};#eta_{#Lambda};<#it{R}>", kTProfile, {axisConfigurations.axisV0EtaCoarse});
      histosRingFamily.add((folder + "/LeadJet/EtaDependence/pRingObservableProxyEta").c_str(), "<#it{R}> vs #eta_{Jet};#eta_{Jet};<#it{R}>", kTProfile, {axisConfigurations.axisEtaCoarse});
      histosRingFamily.add((folder + "/LeadJet/EtaDependence/pRingObservableProxyEtaHighEtaRes").c_str(), "<#it{R}> vs #eta_{Jet};#eta_{Jet};<#it{R}>", kTProfile, {axisConfigurations.axisEta});

      histosRingFamily.add((folder + "/SubJet/EtaDependence/pRingObservableEtaLambda").c_str(), "<#it{R}> vs #eta_{#Lambda} (SubJet);#eta_{#Lambda};<#it{R}>", kTProfile, {axisConfigurations.axisV0EtaCoarse});
      histosRingFamily.add((folder + "/SubJet/EtaDependence/pRingObservableProxyEta").c_str(), "<#it{R}> vs #eta_{SubJet};#eta_{SubJet};<#it{R}>", kTProfile, {axisConfigurations.axisEtaCoarse});

      histosRingFamily.add((folder + "/LeadP/EtaDependence/pRingObservableEtaLambda").c_str(), "<#it{R}> vs #eta_{#Lambda} (LeadP);#eta_{#Lambda};<#it{R}>", kTProfile, {axisConfigurations.axisV0EtaCoarse});
      histosRingFamily.add((folder + "/LeadP/EtaDependence/pRingObservableProxyEta").c_str(), "<#it{R}> vs #eta_{LeadP};#eta_{LeadP};<#it{R}>", kTProfile, {axisConfigurations.axisEtaCoarse});
      histosRingFamily.add((folder + "/LeadP/EtaDependence/pRingObservableProxyEtaHighEtaRes").c_str(), "<#it{R}> vs #eta_{LeadP};#eta_{LeadP};<#it{R}>", kTProfile, {axisConfigurations.axisEta});
      // For the leading particle:
      histosRingFamily.add((folder + "/LeadP/hRingObservableCounts").c_str(), "Counts vs <#it{R}>_{LeadP};<#it{R}>; Counts", kTH1D, {axisConfigurations.axisRingCounts});
      histosRingFamily.add((folder + "/LeadP/pRingObservableDeltaPhi").c_str(), "<#it{R}> vs #Delta#varphi_{LeadP};#Delta#varphi_{LeadP};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      histosRingFamily.add((folder + "/LeadP/pRingObservableDeltaTheta").c_str(), "<#it{R}> vs #Delta#theta_{LeadP};#Delta#theta_{LeadP};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaTheta});
      histosRingFamily.add((folder + "/LeadP/pRingObservableIntegrated").c_str(), "Integrated <#it{R}> (LeadP); ;<#it{R}>", kTProfile, {{1, -0.5, 0.5}});
      histosRingFamily.add((folder + "/LeadP/pRingObservableLambdaPt").c_str(), "<#it{R}> vs #it{p}_{T}^{#Lambda} (LeadP);#it{p}_{T}^{#Lambda} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisPt});
      // For the second-to-leading jet:
      histosRingFamily.add((folder + "/SubJet/hRingObservableCounts").c_str(), "Counts vs <#it{R}>_{SubJet};<#it{R}>; Counts", kTH1D, {axisConfigurations.axisRingCounts});
      histosRingFamily.add((folder + "/SubJet/pRingObservableDeltaPhi").c_str(), "<#it{R}> vs #Delta#varphi_{SubJet};#Delta#varphi_{SubJet};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      histosRingFamily.add((folder + "/SubJet/pRingObservableDeltaTheta").c_str(), "<#it{R}> vs #Delta#theta_{SubJet};#Delta#theta_{SubJet};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaTheta});
      histosRingFamily.add((folder + "/SubJet/pRingObservableIntegrated").c_str(), "Integrated <#it{R}> (SubJet); ;<#it{R}>", kTProfile, {{1, -0.5, 0.5}});
      histosRingFamily.add((folder + "/SubJet/pRingObservableLambdaPt").c_str(), "<#it{R}> vs #it{p}_{T}^{#Lambda} (SubJet);#it{p}_{T}^{#Lambda} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisPt});

      // For the Zvtx dependence:
      histosRingFamily.add((folder + "/LeadJet/pRingObservablePVz").c_str(), "<#it{R}>_{LeadJet} vs PVz;PVz (cm);<#it{R}>", kTProfile, {axisConfigurations.axisPVzCoarse});
      histosRingFamily.add((folder + "/SubJet/pRingObservablePVz").c_str(), "<#it{R}>_{SubLead} vs PVz;PVz (cm);<#it{R}>", kTProfile, {axisConfigurations.axisPVzCoarse});
      histosRingFamily.add((folder + "/LeadP/pRingObservablePVz").c_str(), "<#it{R}>_{LeadP} vs PVz;PVz (cm);<#it{R}>", kTProfile, {axisConfigurations.axisPVzCoarse});
      // ===============================
      // 2D TProfiles (Lambda correlations)
      // ===============================
      histosRingFamily.add((folder + "/LeadJet/p2dRingObservableDeltaPhiVsLambdaPt").c_str(), "<#it{R}> vs #Delta#varphi_{jet} vs #it{p}_{T}^{#Lambda};#Delta#varphi_{jet};#it{p}_{T}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisPt});
      histosRingFamily.add((folder + "/LeadJet/p2dRingObservableDeltaThetaVsLambdaPt").c_str(), "<#it{R}> vs #Delta#theta_{jet} vs #it{p}_{T}^{#Lambda};#Delta#theta_{jet};#it{p}_{T}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisPt});
      // ===============================
      // 2D TProfiles (Jet correlations)
      // ===============================
      histosRingFamily.add((folder + "/LeadJet/p2dRingObservableDeltaPhiVsProxyPt").c_str(), "<#it{R}> vs #Delta#varphi_{jet} vs Lead Jet #it{p}_{T};#Delta#varphi_{jet};#it{p}_{T}^{LeadJet} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisJetPt});
      histosRingFamily.add((folder + "/LeadJet/p2dRingObservableDeltaThetaVsProxyPt").c_str(), "<#it{R}> vs #Delta#theta_{jet} vs Lead Jet #it{p}_{T};#Delta#theta_{jet};#it{p}_{T}^{LeadJet} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisJetPt});

      // ===============================
      // Multi-dimensional histograms for signal extraction
      // (Mass-dependent polarization extraction)
      // ===============================
      // Simple invariant mass plot for QA:
      histosRingFamily.add((folder + "/LeadJet/QA/hMass").c_str(), "#Lambda Mass;m_{p#pi} (GeV/c^{2});Counts", kTH1D, {axisConfigurations.axisLambdaMass});
      histosRingFamily.add((folder + "/LeadJet/hMassSigExtract").c_str(), "#Lambda Mass (Sig Extract);m_{p#pi} (GeV/c^{2});Counts", kTH1D, {axisConfigurations.axisLambdaMassSigExtract});
      // 1D Mass dependence of observable numerator:
      histosRingFamily.add((folder + "/LeadJet/QA/hRingObservableNumMass").c_str(), "Ring Observable Numerator vs Mass;m_{p#pi} (GeV/c^{2});Counts", kTH1D, {axisConfigurations.axisLambdaMassSigExtract});
      // --- 2D counters: Angle vs Mass vs ---
      histosRingFamily.add((folder + "/LeadJet/QA/h2dDeltaPhiVsMass").c_str(), "#Delta#varphi_{jet} vs Mass;#Delta#varphi_{jet};m_{p#pi} (GeV/c^{2})", kTH2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisLambdaMassSigExtract});
      histosRingFamily.add((folder + "/LeadJet/QA/h2dDeltaThetaVsMass").c_str(), "#Delta#theta_{jet} vs Mass;#Delta#theta_{jet};m_{p#pi} (GeV/c^{2})", kTH2D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisLambdaMassSigExtract});
      // --- 3D counters: Angle vs Mass vs Lambda pT ---
      histosRingFamily.add((folder + "/LeadJet/QA/h3dDeltaPhiVsMassVsLambdaPt").c_str(), "#Delta#varphi_{jet} vs Mass vs #it{p}_{T}^{#Lambda};#Delta#varphi_{jet};m_{p#pi} (GeV/c^{2});#it{p}_{T}^{#Lambda} (GeV/c)", kTH3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisPtSigExtract});
      histosRingFamily.add((folder + "/LeadJet/QA/h3dDeltaThetaVsMassVsLambdaPt").c_str(), "#Delta#theta_{jet} vs Mass vs #it{p}_{T}^{#Lambda};#Delta#theta_{jet};m_{p#pi} (GeV/c^{2});#it{p}_{T}^{#Lambda} (GeV/c)", kTH3D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisPtSigExtract});
      // --- 3D counters: Angle vs Mass vs Lead Jet pT ---
      histosRingFamily.add((folder + "/LeadJet/QA/h3dDeltaPhiVsMassVsLeadJetPt").c_str(), "#Delta#varphi_{jet} vs Mass vs Lead Jet #it{p}_{T};#Delta#varphi_{jet};m_{p#pi} (GeV/c^{2});#it{p}_{T}^{LeadJet} (GeV/c)", kTH3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisJetPtSigExtract});
      histosRingFamily.add((folder + "/LeadJet/QA/h3dDeltaThetaVsMassVsLeadJetPt").c_str(), "#Delta#theta_{jet} vs Mass vs Lead Jet #it{p}_{T};#Delta#theta_{jet};m_{p#pi} (GeV/c^{2});#it{p}_{T}^{LeadJet} (GeV/c)", kTH3D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisJetPtSigExtract});

      // ===============================
      // TProfiles vs Mass: quick glancing and signal extraction
      // ===============================
      // TProfile of ring vs mass (integrated in all phi, and properly normalized by N_Lambda):
      histosRingFamily.add((folder + "/LeadJet/pRingObservableMass").c_str(), "<#it{R}> vs Mass;m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
      histosRingFamily.add((folder + "/LeadP/pRingObservableMass").c_str(), "<#it{R}> vs Mass (LeadP);m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
      histosRingFamily.add((folder + "/SubJet/pRingObservableMass").c_str(), "<#it{R}> vs Mass (SubJet);m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile, {axisConfigurations.axisLambdaMassSigExtract});

      // KappaEff moments, per proxy and vs mass for signal extraction:
      // With u the unitless ring (R without its prefactor), v the unitless full ring and w = e \cdot n the projection weight,
      // kappa = 3 <u v>/<w> -- the cross moment, exact for any projection. Full ring: u = v and w = 1. R_z and R_z^T: w = n_z^2.
      for (const std::string& proxy : {std::string("LeadJet"), std::string("LeadP"), std::string("SubJet")}) {
        histosRingFamily.add((folder + "/" + proxy + "/KappaEff/pKappaNumVsMass").c_str(), ("<u v> vs Mass (" + proxy + ");m_{p#pi} (GeV/c^{2});<u v>").c_str(), kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
        histosRingFamily.add((folder + "/" + proxy + "/KappaEff/pKappaDenVsMass").c_str(), ("<w> vs Mass (" + proxy + ");m_{p#pi} (GeV/c^{2});<w>").c_str(), kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
        histosRingFamily.add((folder + "/" + proxy + "/KappaEff/pKappaNumTimesDenVsMass").c_str(), ("<u v w> vs Mass (" + proxy + ");m_{p#pi} (GeV/c^{2});<u v w>").c_str(), kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
        histosRingFamily.add((folder + "/" + proxy + "/KappaEff/pRingTimesDenVsMass").c_str(), ("<#it{R} w> vs Mass (" + proxy + ");m_{p#pi} (GeV/c^{2});<#it{R} w>").c_str(), kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
        // The same moments per CheapSigExtract region, from v0InMassWindow and v0InMassPeak: bin 1 sideband, bin 2 peak, underflow otherwise
        histosRingFamily.add((folder + "/" + proxy + "/KappaEff/pRingVsMassRegion").c_str(), ("<#it{R}> per mass region (" + proxy + ");mass region (1 sideband, 2 peak);<#it{R}>").c_str(), kTProfile, {{2, 0., 2.}});
        histosRingFamily.add((folder + "/" + proxy + "/KappaEff/pKappaNumVsMassRegion").c_str(), ("<u v> per mass region (" + proxy + ");mass region (1 sideband, 2 peak);<u v>").c_str(), kTProfile, {{2, 0., 2.}});
        histosRingFamily.add((folder + "/" + proxy + "/KappaEff/pKappaDenVsMassRegion").c_str(), ("<w> per mass region (" + proxy + ");mass region (1 sideband, 2 peak);<w>").c_str(), kTProfile, {{2, 0., 2.}});
        histosRingFamily.add((folder + "/" + proxy + "/KappaEff/pKappaNumTimesDenVsMassRegion").c_str(), ("<u v w> per mass region (" + proxy + ");mass region (1 sideband, 2 peak);<u v w>").c_str(), kTProfile, {{2, 0., 2.}});
        histosRingFamily.add((folder + "/" + proxy + "/KappaEff/pRingTimesDenVsMassRegion").c_str(), ("<#it{R} w> per mass region (" + proxy + ");mass region (1 sideband, 2 peak);<#it{R} w>").c_str(), kTProfile, {{2, 0., 2.}});

        // Differential KappaEff -- The same moments per proxy pT and per centrality, still per mass region:
        // Axes are kept the same as the event mixing axes. Otherwise, a borrowed proxy could land in another bin than its target,
        // so Data and MixedEv would no longer share their Lambdas bin by bin, and the subtraction of the mixed-event fake from Data would be off in post-processing.
        // (the pT is the proxy the event mixing source's, which sits in the target's own pT bin by construction)
        struct DiffMoment { std::string name, title; };
        struct DiffVariable { std::string name, title; AxisSpec axis; };
        const std::vector<DiffMoment> diffMoments = {{"Ring", "<#it{R}>"}, {"KappaNum", "<u v>"}, {"KappaDen", "<w>"}, {"KappaNumTimesDen", "<u v w>"}, {"RingTimesDen", "<#it{R} w>"}};
        const std::vector<DiffVariable> diffVariables = {{"ProxyPt", "#it{p}_{T}^{" + proxy + "} (GeV/c)", axisConfigurations.axisJetPt}, {"Centrality", "Centrality (%)", axisConfigurations.axisCentrality}};
        for (const auto& mom : diffMoments) {
          for (const auto& var : diffVariables) {
            histosRingFamily.add((folder + "/" + proxy + "/KappaEff/Differential/p2d" + mom.name + "Vs" + var.name + "VsMassRegion").c_str(),
                                 (mom.title + " vs " + var.name + " per mass region (" + proxy + ");" + var.title + ";mass region (1 sideband, 2 peak);" + mom.title).c_str(),
                                 kTProfile2D, {var.axis, {2, 0., 2.}});
          }
        }
      }
      // TProfile2D: <R> vs Mass (DeltaPhi)
      histosRingFamily.add((folder + "/LeadJet/p2dRingObservableDeltaPhiVsMass").c_str(), "<#it{R}> vs #Delta#varphi_{jet} vs Mass;#Delta#varphi_{jet};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisLambdaMassSigExtract});
      // TProfile2D: <R> vs Mass (DeltaTheta)
      histosRingFamily.add((folder + "/LeadJet/p2dRingObservableDeltaThetaVsMass").c_str(), "<#it{R}> vs #Delta#theta_{jet} vs Mass;#Delta#theta_{jet};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisLambdaMassSigExtract});
      // TProfile2D: <R> vs Mass (EtaLambda):
      histosRingFamily.add((folder + "/LeadJet/p2dRingObservableEtaLambdaVsMass").c_str(), "<#it{R}> vs #eta_{#Lambda} vs Mass;#eta_{#Lambda};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisLambdaMassSigExtract});
      // TProfile2D: <R> vs EtaProxy Vs Mass
      histosRingFamily.add((folder + "/LeadJet/p2dRingObservableProxyEtaVsMass").c_str(), "<#it{R}> vs #eta_{jet} vs Mass;#eta_{jet};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
      histosRingFamily.add((folder + "/LeadP/p2dRingObservableProxyEtaVsMass").c_str(), "<#it{R}> vs #eta_{LeadP} vs Mass;#eta_{LeadP};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
      histosRingFamily.add((folder + "/SubJet/p2dRingObservableProxyEtaVsMass").c_str(), "<#it{R}> vs #eta_{SubJet} vs Mass;#eta_{SubJet};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
      // Auxiliary counters for EtaProxy Vs Mass:
      histosRingFamily.add((folder + "/LeadJet/h2dCounterProxyEtaVsMass").c_str(), "V0 Counter vs #eta_{jet} vs Mass;#eta_{jet};m_{p#pi} (GeV/c^{2});Counter", kTH2D, {axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
      histosRingFamily.add((folder + "/LeadP/h2dCounterProxyEtaVsMass").c_str(), "V0 Counter vs #eta_{LeadP} vs Mass;#eta_{LeadP};m_{p#pi} (GeV/c^{2});Counter", kTH2D, {axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
      histosRingFamily.add((folder + "/SubJet/h2dCounterProxyEtaVsMass").c_str(), "V0 Counter vs #eta_{SubJet} vs Mass;#eta_{SubJet};m_{p#pi} (GeV/c^{2});Counter", kTH2D, {axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});

      // TProfile2D: <R> vs Eta Lambda vs Eta Jet (Understanding eta dependence seen in pRingEtaCuts)
      histosRingFamily.add((folder + "/LeadJet/EtaDependence/hCounterEtaLambdaMinusProxyEta").c_str(), "N_{V0s} vs #eta_{#Lambda} - #eta_{Jet};#eta_{#Lambda} - #eta_{Jet}; N_{V0s}", kTH1D, {axisConfigurations.axisDeltaEtaCoarse});
      histosRingFamily.add((folder + "/LeadJet/EtaDependence/pRingObservableEtaLambdaMinusProxyEta").c_str(), "<#it{R}> vs #eta_{#Lambda} - #eta_{Jet};#eta_{#Lambda} - #eta_{Jet};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaEtaCoarse});
      histosRingFamily.add((folder + "/LeadJet/EtaDependence/p2dRingObservableEtaLambdaVsProxyEta").c_str(), "<#it{R}> vs #eta_{#Lambda} vs #eta_{Jet};#eta_{#Lambda};#eta_{Jet};<#it{R}>", kTProfile2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisEtaCoarse});
      histosRingFamily.add((folder + "/LeadJet/EtaDependence/p2dRingObservableEtaLambdaVsProxyEta_FineBins").c_str(), "<#it{R}> vs #eta_{#Lambda} vs #eta_{Jet} (fine bins);#eta_{#Lambda};#eta_{Jet};<#it{R}>", kTProfile2D, {axisConfigurations.axisV0Eta, axisConfigurations.axisEta});
      histosRingFamily.add((folder + "/LeadP/EtaDependence/p2dRingObservableEtaLambdaVsProxyEta").c_str(), "<#it{R}> vs #eta_{#Lambda} vs #eta_{LeadP};#eta_{#Lambda};#eta_{LeadP};<#it{R}>", kTProfile2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisEtaCoarse});
      histosRingFamily.add((folder + "/SubJet/EtaDependence/p2dRingObservableEtaLambdaVsProxyEta").c_str(), "<#it{R}> vs #eta_{#Lambda} vs #eta_{SubJet};#eta_{#Lambda};#eta_{SubJet};<#it{R}>", kTProfile2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisEtaCoarse});
      // Counters for these histograms, instead of only TProfile2Ds:
      histosRingFamily.add((folder + "/LeadJet/EtaDependence/h2dCounterEtaLambdaVsProxyEta").c_str(), "Counts, #eta_{#Lambda} vs #eta_{Jet};#eta_{#Lambda};#eta_{Jet};Counts", kTH2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisEtaCoarse});
      histosRingFamily.add((folder + "/LeadJet/EtaDependence/h2dCounterEtaLambdaVsProxyEta_FineBins").c_str(), "Counts (fine bins), #eta_{#Lambda} vs #eta_{Jet};#eta_{#Lambda};#eta_{Jet};Counts", kTH2D, {axisConfigurations.axisV0Eta, axisConfigurations.axisEta});
      histosRingFamily.add((folder + "/LeadP/EtaDependence/h2dCounterEtaLambdaVsProxyEta").c_str(), "Counts, #eta_{#Lambda} vs #eta_{LeadP};#eta_{#Lambda};#eta_{LeadP};Counts", kTH2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisEtaCoarse});
      histosRingFamily.add((folder + "/SubJet/EtaDependence/h2dCounterEtaLambdaVsProxyEta").c_str(), "Counts, #eta_{#Lambda} vs #eta_{SubJet};#eta_{#Lambda};#eta_{SubJet};Counts", kTH2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisEtaCoarse});
      // --- TProfile3D: <R> vs DeltaPhi vs Mass vs LambdaPt ---
      histosRingFamily.add((folder + "/LeadJet/p3dRingObservableDeltaPhiVsMassVsLambdaPt").c_str(), "<#it{R}> vs #Delta#varphi_{jet} vs Mass vs #it{p}_{T}^{#Lambda};#Delta#varphi_{jet};m_{p#pi} (GeV/c^{2});#it{p}_{T}^{#Lambda} (GeV/c)", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisPtSigExtract});
      // --- TProfile3D: <R> vs DeltaTheta vs Mass vs LambdaPt ---
      histosRingFamily.add((folder + "/LeadJet/p3dRingObservableDeltaThetaVsMassVsLambdaPt").c_str(), "<#it{R}> vs #Delta#theta_{jet} vs Mass vs #it{p}_{T}^{#Lambda};#Delta#theta_{jet};m_{p#pi} (GeV/c^{2});#it{p}_{T}^{#Lambda} (GeV/c)", kTProfile3D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisPtSigExtract});
      // --- TProfile3D: <R> vs DeltaPhi vs Mass vs LeadJetPt ---
      histosRingFamily.add((folder + "/LeadJet/p3dRingObservableDeltaPhiVsMassVsProxyPt").c_str(), "<#it{R}> vs #Delta#varphi_{jet} vs Mass vs Lead Jet #it{p}_{T};#Delta#varphi_{jet};m_{p#pi} (GeV/c^{2});#it{p}_{T}^{LeadJet} (GeV/c)", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisJetPtSigExtract});
      // --- TProfile3D: <R> vs DeltaTheta vs Mass vs LeadJetPt ---
      histosRingFamily.add((folder + "/LeadJet/p3dRingObservableDeltaThetaVsMassVsProxyPt").c_str(), "<#it{R}> vs #Delta#theta_{jet} vs Mass vs Lead Jet #it{p}_{T};#Delta#theta_{jet};m_{p#pi} (GeV/c^{2});#it{p}_{T}^{LeadJet} (GeV/c)", kTProfile3D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisJetPtSigExtract});

      // ===============================
      // Mass histograms with centrality
      // ===============================
      // Counters
      histosRingFamily.add((folder + "/LeadJet/QA/h3dDeltaPhiVsMassVsCent").c_str(), "#Delta#varphi_{jet} vs Mass vs Centrality;#Delta#varphi_{jet};m_{p#pi} (GeV/c^{2});Centrality (%)", kTH3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisCentrality});
      histosRingFamily.add((folder + "/LeadJet/QA/h3dDeltaThetaVsMassVsCent").c_str(), "#Delta#theta_{jet} vs Mass vs Centrality;#Delta#theta_{jet};m_{p#pi} (GeV/c^{2});Centrality (%)", kTH3D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisCentrality});
      // Useful TProfiles:
      // --- TProfile1D: Integrated <R> vs Centrality:
      histosRingFamily.add((folder + "/LeadJet/pRingVsCentrality").c_str(), "<#it{R}> vs Centrality;Centrality (%);<#it{R}>", kTProfile, {axisConfigurations.axisCentrality});
      // --- TProfile2D: <R> vs Mass vs Centrality ---
      histosRingFamily.add((folder + "/LeadJet/p2dRingObservableMassVsCent").c_str(), "<#it{R}> vs Mass vs Centrality;m_{p#pi} (GeV/c^{2});Centrality (%);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisCentrality});
      // --- TProfile3D: <R> vs DeltaPhi vs Mass vs Centrality ---
      histosRingFamily.add((folder + "/LeadJet/p3dRingObservableDeltaPhiVsMassVsCent").c_str(), "<#it{R}> vs #Delta#varphi_{jet} vs Mass vs Centrality;#Delta#varphi_{jet};m_{p#pi} (GeV/c^{2});Centrality (%)", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisCentrality});
      // --- TProfile3D: <R> vs DeltaTheta vs Mass vs Centrality ---
      histosRingFamily.add((folder + "/LeadJet/p3dRingObservableDeltaThetaVsMassVsCent").c_str(), "<#it{R}> vs #Delta#theta_{jet} vs Mass vs Centrality;#Delta#theta_{jet};m_{p#pi} (GeV/c^{2});Centrality (%)", kTProfile3D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisCentrality});

      // ===============================
      // QA histograms - Useful numbers
      // ===============================
      // (TODO: implement these!)
      // (TODO: implement momentum imbalance checks for jets!)
      // Added to a separate folder for further control (changed the usage of the "folder" string):
      // histosRingFamily.add(("QA_Numbers/" + folder + "/hValidLeadJets").c_str(), "hValidLeadJets", kTH1D, {{1,0,1}});
      // TODO: Add "frequency of jets per pT" histograms either here or in the TableProducer

      // Estimating error bars with the Delta Method for <R> = A/B:
      // 1D Delta Method for Integrated observable:
      histosRingFamily.add((folder + "/DeltaMethod/hIntegrated").c_str(), "Delta Method Accumulators Integrated;Component;Counts", kTH1D, {axisConfigurations.axisDeltaComponents});

      // 2D Delta Method for Differentials
      histosRingFamily.add((folder + "/DeltaMethod/h2dDeltaThetaVsDeltaComp").c_str(), "Delta Method vs #Delta#theta_{jet};#Delta#theta_{jet};Component", kTH2D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisDeltaComponents});
      histosRingFamily.add((folder + "/DeltaMethod/h2dLambdaPtVsDeltaComp").c_str(), "Delta Method vs #Lambda #it{p}_{T};#it{p}_{T}^{#Lambda} (GeV/c);Component", kTH2D, {axisConfigurations.axisPt, axisConfigurations.axisDeltaComponents});
      histosRingFamily.add((folder + "/DeltaMethod/h2dMassVsDeltaComp").c_str(), "Delta Method vs Mass;m_{p#pi} (GeV/c^{2});Component", kTH2D, {axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisDeltaComponents});
    };
    // Execute local lambda to register histogram families:
    if (familySwitches.doFamilyRing)
      addRingObservableFamily("Ring");
    if (familySwitches.doFamilyRingKinematicCuts)
      addRingObservableFamily("RingKinematicCuts");
    if (familySwitches.doFamilyJetKinematicCuts)
      addRingObservableFamily("JetKinematicCuts");
    if (familySwitches.doFamilyJetAndLambdaKinematicCuts)
      addRingObservableFamily("JetAndLambdaKinematicCuts");

    // R_z-only diagnostics -- Compares the 3D ring, the projected ring and other useful diagnostics:
    if (useRingZ || useRingZT) {
      histos.add("RzDiagnostics/LeadJet/pRingVsChi", "<#it{R}_{z}> vs #chi (LeadJet) (#hat{n} = cos(#chi)#hat{#phi} + sin(#chi)#hat{#theta});#chi_{LeadJet};<#it{R}_{z}>", kTProfile, {axisConfigurations.axisDeltaPhi}); // chi is the angle such that n_hat = cos(chi) phi_hat + sin(chi) theta_hat.
      histos.add("RzDiagnostics/LeadP/pRingVsChi", "<#it{R}_{z}> vs #chi (LeadP) (#hat{n} = cos(#chi)#hat{#phi} + sin(#chi)#hat{#theta});#chi_{LeadP};<#it{R}_{z}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("RzDiagnostics/SubJet/pRingVsChi", "<#it{R}_{z}> vs #chi (SubJet) (#hat{n} = cos(#chi)#hat{#phi} + sin(#chi)#hat{#theta});#chi_{SubJet};<#it{R}_{z}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("RzDiagnostics/LeadJet/p2dRingVsChiVsMass", "<#it{R}_{z}> vs #chi vs Mass (LeadJet) (#hat{n} = cos(#chi)#hat{#phi} + sin(#chi)#hat{#theta});#chi_{LeadJet};m_{p#pi} (GeV/c^{2});<#it{R}_{z}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("RzDiagnostics/LeadP/p2dRingVsChiVsMass", "<#it{R}_{z}> vs #chi vs Mass (LeadP) (#hat{n} = cos(#chi)#hat{#phi} + sin(#chi)#hat{#theta});#chi_{LeadP};m_{p#pi} (GeV/c^{2});<#it{R}_{z}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("RzDiagnostics/SubJet/p2dRingVsChiVsMass", "<#it{R}_{z}> vs #chi vs Mass (SubJet) (#hat{n} = cos(#chi)#hat{#phi} + sin(#chi)#hat{#theta});#chi_{SubJet};m_{p#pi} (GeV/c^{2});<#it{R}_{z}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("RzDiagnostics/LeadJet/p2dRingVsDeltaPhiVsDeltaEta", "<#it{R}_{z}> vs (#Delta#varphi_{jet}, #Delta#eta_{jet});#Delta#varphi_{jet}=#phi_{#Lambda} - #phi_{Jet};#eta_{#Lambda}-#eta_{Jet};<#it{R}_{z}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisDeltaEtaCoarse});
      // R_perp = R - R_z, filled directly to store the proper error propagation (R and R_z share every candidate, thus are correlated):
      histos.add("RzDiagnostics/pRingPerpIntegrated", "<#it{R}_{#perp}> per proxy; ;<#it{R}_{#perp}> = <#it{R}> - <#it{R}_{z}>", kTProfile, {{3, 0, 3}});
      histos.get<TProfile>(HIST("RzDiagnostics/pRingPerpIntegrated"))->GetXaxis()->SetBinLabel(1, "LeadJet");
      histos.get<TProfile>(HIST("RzDiagnostics/pRingPerpIntegrated"))->GetXaxis()->SetBinLabel(2, "LeadP");
      histos.get<TProfile>(HIST("RzDiagnostics/pRingPerpIntegrated"))->GetXaxis()->SetBinLabel(3, "SubJet");
      histos.add("RzDiagnostics/LeadJet/pRingPerpVsMass", "<#it{R}_{#perp}> vs Mass (LeadJet);m_{p#pi} (GeV/c^{2});<#it{R}_{#perp}> = <#it{R}> - <#it{R}_{z}>", kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
      histos.add("RzDiagnostics/LeadP/pRingPerpVsMass", "<#it{R}_{#perp}> vs Mass (LeadP);m_{p#pi} (GeV/c^{2});<#it{R}_{#perp}> = <#it{R}> - <#it{R}_{z}>", kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
      histos.add("RzDiagnostics/SubJet/pRingPerpVsMass", "<#it{R}_{#perp}> vs Mass (SubJet);m_{p#pi} (GeV/c^{2});<#it{R}_{#perp}> = <#it{R}> - <#it{R}_{z}>", kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
      histos.add("RzDiagnostics/LeadJet/pRingPerpVsPhiAEE", "<#it{R}_{#perp}> vs AEE angle (LeadJet);#phi_{#Lambda-like}-#phi_{p-like}^{*};<#it{R}_{#perp}> = <#it{R}> - <#it{R}_{z}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("RzDiagnostics/LeadP/pRingPerpVsPhiAEE", "<#it{R}_{#perp}> vs AEE angle (LeadP);#phi_{#Lambda-like}-#phi_{p-like}^{*};<#it{R}_{#perp}> = <#it{R}> - <#it{R}_{z}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("RzDiagnostics/SubJet/pRingPerpVsPhiAEE", "<#it{R}_{#perp}> vs AEE angle (SubJet);#phi_{#Lambda-like}-#phi_{p-like}^{*};<#it{R}_{#perp}> = <#it{R}> - <#it{R}_{z}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("RzDiagnostics/LeadJet/pRingPerpVsDeltaPhi", "<#it{R}_{#perp}> vs #Delta#varphi (LeadJet);#Delta#varphi_{jet};<#it{R}_{#perp}> = <#it{R}> - <#it{R}_{z}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("RzDiagnostics/LeadP/pRingPerpVsDeltaPhi", "<#it{R}_{#perp}> vs #Delta#varphi (LeadP);#Delta#varphi_{LeadP};<#it{R}_{#perp}> = <#it{R}> - <#it{R}_{z}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("RzDiagnostics/SubJet/pRingPerpVsDeltaPhi", "<#it{R}_{#perp}> vs #Delta#varphi (SubJet);#Delta#varphi_{SubJet};<#it{R}_{#perp}> = <#it{R}> - <#it{R}_{z}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      // Standard (unweighted) longitudinal acceptance factor, on the LeadJet sample, to compare against the n_z^2-weighted kappa_z:
      // Helicity term, R_z - R_z^T = (3/alpha)(p* \cdot p_Lambda) p_Lambda,z n_z
      // (measures what R_z^T removes from R_z)
      histos.add("RzDiagnostics/pHelicityTermIntegrated", "Helicity term #it{R}_{z} - #it{R}_{z}^{T} per proxy; ;<#it{R}_{z} - #it{R}_{z}^{T}>", kTProfile, {{3, 0, 3}});
      histos.get<TProfile>(HIST("RzDiagnostics/pHelicityTermIntegrated"))->GetXaxis()->SetBinLabel(1, "LeadJet");
      histos.get<TProfile>(HIST("RzDiagnostics/pHelicityTermIntegrated"))->GetXaxis()->SetBinLabel(2, "LeadP");
      histos.get<TProfile>(HIST("RzDiagnostics/pHelicityTermIntegrated"))->GetXaxis()->SetBinLabel(3, "SubJet");
      histos.add("RzDiagnostics/LeadJet/pHelicityTermVsEtaLambda", "Helicity term vs #eta_{#Lambda} (LeadJet);#eta_{#Lambda};<#it{R}_{z} - #it{R}_{z}^{T}>", kTProfile, {axisConfigurations.axisV0EtaCoarse});
      histos.add("RzDiagnostics/LeadP/pHelicityTermVsEtaLambda", "Helicity term vs #eta_{#Lambda} (LeadP);#eta_{#Lambda};<#it{R}_{z} - #it{R}_{z}^{T}>", kTProfile, {axisConfigurations.axisV0EtaCoarse});
      histos.add("RzDiagnostics/SubJet/pHelicityTermVsEtaLambda", "Helicity term vs #eta_{#Lambda} (SubJet);#eta_{#Lambda};<#it{R}_{z} - #it{R}_{z}^{T}>", kTProfile, {axisConfigurations.axisV0EtaCoarse});
      histos.add("RzDiagnostics/LeadJet/p2dHelicityTermVsDeltaPhiVsMass", "Helicity term vs (#Delta#varphi_{jet}, mass);#Delta#varphi_{jet};m_{p#pi} (GeV/c^{2});<#it{R}_{z} - #it{R}_{z}^{T}>", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisLambdaMassSigExtract});
      // The ring geometry vs eta_Lambda: <n_z> is the proxy-side balance, <n_z^2> the signal weight (note, Test Case 5.3)
      histos.add("RzDiagnostics/LeadJet/pNzVsEtaLambda", "<n_{z}> vs #eta_{#Lambda} (LeadJet);#eta_{#Lambda};<n_{z}>", kTProfile, {axisConfigurations.axisV0EtaCoarse});
      histos.add("RzDiagnostics/LeadP/pNzVsEtaLambda", "<n_{z}> vs #eta_{#Lambda} (LeadP);#eta_{#Lambda};<n_{z}>", kTProfile, {axisConfigurations.axisV0EtaCoarse});
      histos.add("RzDiagnostics/SubJet/pNzVsEtaLambda", "<n_{z}> vs #eta_{#Lambda} (SubJet);#eta_{#Lambda};<n_{z}>", kTProfile, {axisConfigurations.axisV0EtaCoarse});
      histos.add("RzDiagnostics/LeadJet/pNzSqVsEtaLambda", "<n_{z}^{2}> vs #eta_{#Lambda} (LeadJet);#eta_{#Lambda};<n_{z}^{2}>", kTProfile, {axisConfigurations.axisV0EtaCoarse});
      histos.add("RzDiagnostics/LeadP/pNzSqVsEtaLambda", "<n_{z}^{2}> vs #eta_{#Lambda} (LeadP);#eta_{#Lambda};<n_{z}^{2}>", kTProfile, {axisConfigurations.axisV0EtaCoarse});
      histos.add("RzDiagnostics/SubJet/pNzSqVsEtaLambda", "<n_{z}^{2}> vs #eta_{#Lambda} (SubJet);#eta_{#Lambda};<n_{z}^{2}>", kTProfile, {axisConfigurations.axisV0EtaCoarse});
      // The sideband-sine test: the ring vs (Delta phi, mass), split by the A/C hemisphere of the Lambda
      histos.add("RzDiagnostics/LeadJet/p2dRingVsDeltaPhiVsMassEtaPos", "<#it{R}_{z}> vs (#Delta#varphi_{jet}, mass), #eta_{#Lambda} > 0;#Delta#varphi_{jet};m_{p#pi} (GeV/c^{2});<#it{R}_{z}>", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("RzDiagnostics/LeadJet/p2dRingVsDeltaPhiVsMassEtaNeg", "<#it{R}_{z}> vs (#Delta#varphi_{jet}, mass), #eta_{#Lambda} < 0;#Delta#varphi_{jet};m_{p#pi} (GeV/c^{2});<#it{R}_{z}>", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("RzDiagnostics/LeadJet/pCosSqThetaStarZVsMass", "<cos^{2}#theta^{*}_{z}> vs Mass (LeadJet);m_{p#pi} (GeV/c^{2});<cos^{2}#theta^{*}_{z}>", kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
    }

    // Kinematic QA histograms for diagnosing detector defficiencies (V0s and daughters only. Jets are under JetKinematicsQA):
    histos.add("KinematicsQA/V0/hV0Phi", "hV0Phi; #phi_{V0}; Counts", kTH1D, {axisConfigurations.axisPhi});
    histos.add("KinematicsQA/V0/hV0Eta", "hV0Eta; #eta_{V0}; Counts", kTH1D, {axisConfigurations.axisV0Eta});
    histos.add("KinematicsQA/V0/hV0Pt", "hV0Pt; p_{T, V0}; Counts", kTH1D, {axisConfigurations.axisPt});
    histos.add("KinematicsQA/V0/h2dV0PhiVsEta", "h2dV0PhiVsEta; #phi_{V0}; #eta_{V0}; Counts", kTH2D, {axisConfigurations.axisPhi, axisConfigurations.axisV0Eta});
    histos.add("KinematicsQA/V0/h2dV0PhiVsLeadJetPhi", "h2dV0PhiVsLeadJetPhi; #phi_{V0}; #phi_{Jet}; Counts", kTH2D, {axisConfigurations.axisPhi, axisConfigurations.axisPhi});
    histos.add("KinematicsQA/V0/h2dV0EtaVsLeadJetEta", "h2dV0EtaVsLeadJetEta; #eta_{V0}; #eta_{Jet}; Counts", kTH2D, {axisConfigurations.axisV0Eta, axisConfigurations.axisV0Eta});
    // Pion or proton tracks:
    histos.add("KinematicsQA/V0dau/hPrPhi", "hPrPhi; #phi_{Pr}; Counts", kTH1D, {axisConfigurations.axisPhi});
    histos.add("KinematicsQA/V0dau/hPrEta", "hPrEta; #eta_{Pr}; Counts", kTH1D, {axisConfigurations.axisV0Eta});
    histos.add("KinematicsQA/V0dau/hPrPt", "hPrPt; p_{T, Pr}; Counts", kTH1D, {axisConfigurations.axisPt});
    histos.add("KinematicsQA/V0dau/h2dPrPhiVsEta", "h2dPrPhiVsEta; #phi_{Pr}; #eta_{Pr}; Counts", kTH2D, {axisConfigurations.axisPhi, axisConfigurations.axisV0Eta});
    histos.add("KinematicsQA/V0dau/hPiPhi", "hPiPhi; #phi_{Pi}; Counts", kTH1D, {axisConfigurations.axisPhi});
    histos.add("KinematicsQA/V0dau/hPiEta", "hPiEta; #eta_{Pi}; Counts", kTH1D, {axisConfigurations.axisV0Eta});
    histos.add("KinematicsQA/V0dau/hPiPt", "hPiPt; p_{T, Pi}; Counts", kTH1D, {axisConfigurations.axisPt});
    histos.add("KinematicsQA/V0dau/h2dPiPhiVsEta", "h2dPiPhiVsEta; #phi_{Pi}; #eta_{Pi}; Counts", kTH2D, {axisConfigurations.axisPhi, axisConfigurations.axisV0Eta});
    // Pion-proton track 2Ds Vs V0:
    histos.add("KinematicsQA/V0dau/h2dV0PhiVsPrPhi", "h2dV0PhiVsPrPhi; #phi_{V0}; #phi_{Pr}; Counts", kTH2D, {axisConfigurations.axisPhi, axisConfigurations.axisPhi});
    histos.add("KinematicsQA/V0dau/h2dV0EtaVsPrEta", "h2dV0EtaVsPrEta; #eta_{V0}; #eta_{Pr}; Counts", kTH2D, {axisConfigurations.axisV0Eta, axisConfigurations.axisV0Eta});
    histos.add("KinematicsQA/V0dau/h2dV0PhiVsPiPhi", "h2dV0PhiVsPiPhi; #phi_{V0}; #phi_{Pi}; Counts", kTH2D, {axisConfigurations.axisPhi, axisConfigurations.axisPhi});
    histos.add("KinematicsQA/V0dau/h2dV0EtaVsPiEta", "h2dV0EtaVsPiEta; #eta_{V0}; #eta_{Pi}; Counts", kTH2D, {axisConfigurations.axisV0Eta, axisConfigurations.axisV0Eta});
    // Pion-proton track 2Ds Vs Each Other:
    histos.add("KinematicsQA/V0dau/h2dPrPhiVsPiPhi", "h2dPrPhiVsPiPhi; #phi_{Pr}; #phi_{Pi}; Counts", kTH2D, {axisConfigurations.axisPhi, axisConfigurations.axisPhi});
    histos.add("KinematicsQA/V0dau/h2dPrEtaVsPiEta", "h2dPrEtaVsPiEta; #eta_{Pr}; #eta_{Pi}; Counts", kTH2D, {axisConfigurations.axisV0Eta, axisConfigurations.axisV0Eta});
    // ProtonStar Vs V0:
    histos.add("KinematicsQA/ProtonStar/hPrStarPhi", "hPrStarPhi; #phi_{Pr}^{*}; Counts", kTH1D, {axisConfigurations.axisPhi});
    histos.add("KinematicsQA/ProtonStar/hPrStarEta", "hPrStarEta; #eta_{Pr}^{*}; Counts", kTH1D, {axisConfigurations.axisV0Eta});
    histos.add("KinematicsQA/ProtonStar/hPrStarPt", "hPrStarPt; p_{T, Pr}^{*}; Counts", kTH1D, {axisConfigurations.axisPt});
    histos.add("KinematicsQA/ProtonStar/h2dPrStarPhiVsEta", "h2dPrStarPhiVsEta; #phi_{Pr}^{*}; #eta_{Pr}^{*}; Counts", kTH2D, {axisConfigurations.axisPhi, axisConfigurations.axisV0Eta});
    histos.add("KinematicsQA/ProtonStar/h2dV0PhiVsPrStarPhi", "h2dV0PhiVsPrStarPhi; #phi_{V0}; #phi_{Pr}^{*}; Counts", kTH2D, {axisConfigurations.axisPhi, axisConfigurations.axisPhi});
    histos.add("KinematicsQA/ProtonStar/h2dV0EtaVsPrStarEta", "h2dV0EtaVsPrStarEta; #eta_{V0}; #eta_{Pr}^{*}; Counts", kTH2D, {axisConfigurations.axisV0Eta, axisConfigurations.axisV0Eta});

    histos.add("IntegratedCuts/LeadJet/pRingCuts", "pRingCuts; ;<#it{R}>", kTProfile, {{4, 0, 4}});
    histos.get<TProfile>(HIST("IntegratedCuts/LeadJet/pRingCuts"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
    histos.get<TProfile>(HIST("IntegratedCuts/LeadJet/pRingCuts"))->GetXaxis()->SetBinLabel(2, "p_{T}^{#Lambda}@[0.5,1.5],|y_{#Lambda}|<0.5"); // (v0pt > 0.5 && v0pt < 1.5) && std::abs(lambdaRapidity) < 0.5;
    histos.get<TProfile>(HIST("IntegratedCuts/LeadJet/pRingCuts"))->GetXaxis()->SetBinLabel(3, "|Jet_{#eta}|<0.5");
    histos.get<TProfile>(HIST("IntegratedCuts/LeadJet/pRingCuts"))->GetXaxis()->SetBinLabel(4, "#Lambda + Jet cuts");

    // Same for subleading jet and leading particle:
    histos.add("IntegratedCuts/SubJet/pRingCuts", "pRingCuts; ;<#it{R}>", kTProfile, {{4, 0, 4}});
    histos.get<TProfile>(HIST("IntegratedCuts/SubJet/pRingCuts"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
    histos.get<TProfile>(HIST("IntegratedCuts/SubJet/pRingCuts"))->GetXaxis()->SetBinLabel(2, "p_{T,#Lambda}@[0.5,1.5],|y_{#Lambda}|<0.5");
    histos.get<TProfile>(HIST("IntegratedCuts/SubJet/pRingCuts"))->GetXaxis()->SetBinLabel(3, "|SubJet_{#eta}|<0.5");
    histos.get<TProfile>(HIST("IntegratedCuts/SubJet/pRingCuts"))->GetXaxis()->SetBinLabel(4, "#Lambda + SubJet cuts");

    histos.add("IntegratedCuts/LeadP/pRingCuts", "pRingCuts; ;<#it{R}>", kTProfile, {{4, 0, 4}});
    histos.get<TProfile>(HIST("IntegratedCuts/LeadP/pRingCuts"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
    histos.get<TProfile>(HIST("IntegratedCuts/LeadP/pRingCuts"))->GetXaxis()->SetBinLabel(2, "p_{T}^{#Lambda}@[0.5,1.5],|y_{#Lambda}|<0.5");
    histos.get<TProfile>(HIST("IntegratedCuts/LeadP/pRingCuts"))->GetXaxis()->SetBinLabel(3, "|LeadP_{#eta}|<0.5");
    histos.get<TProfile>(HIST("IntegratedCuts/LeadP/pRingCuts"))->GetXaxis()->SetBinLabel(4, "#Lambda + LeadP cuts");

    // QA for resampling of events -- Influence of jet-related and Lambda-related error bars:
    // The resampling passes share their Lambdas, so the scatter of the pass means against the error of the pass itself separates
    // the jet-sampling variance from the Lambda-common one (the part nProxyResamples cannot average down):
    if (fakePolSwitches.nProxyResamples > 1) {
      const int nPasses = fakePolSwitches.nProxyResamples;
      histos.add("IntegratedCuts/LeadJet/pRingVsResamplingPass", "pRingVsResamplingPass;Resampling pass;<#it{R}>", kTProfile, {{nPasses, -0.5, nPasses - 0.5}});
      histos.add("IntegratedCuts/SubJet/pRingVsResamplingPass", "pRingVsResamplingPass;Resampling pass;<#it{R}>", kTProfile, {{nPasses, -0.5, nPasses - 0.5}});
      histos.add("IntegratedCuts/LeadP/pRingVsResamplingPass", "pRingVsResamplingPass;Resampling pass;<#it{R}>", kTProfile, {{nPasses, -0.5, nPasses - 0.5}});
    }

    // Mass-selected (not properly signal-extracted yet) TProfiles:
    histos.add("IntegratedCuts/LeadJet/p2dRingCutsV0MassPeak", "p2dRingCuts V0MassPeak; ; SidebandWindow (0) or InPeak (1);<#it{R}>", kTProfile2D, {{4, 0, 4}, {2, 0, 2}});
    histos.get<TProfile2D>(HIST("IntegratedCuts/LeadJet/p2dRingCutsV0MassPeak"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
    histos.get<TProfile2D>(HIST("IntegratedCuts/LeadJet/p2dRingCutsV0MassPeak"))->GetXaxis()->SetBinLabel(2, "p_{T}^{#Lambda}@[0.5,1.5],|y_{#Lambda}|<0.5"); // (v0pt > 0.5 && v0pt < 1.5) && std::abs(lambdaRapidity) < 0.5;
    histos.get<TProfile2D>(HIST("IntegratedCuts/LeadJet/p2dRingCutsV0MassPeak"))->GetXaxis()->SetBinLabel(3, "|Jet_{#eta}|<0.5");
    histos.get<TProfile2D>(HIST("IntegratedCuts/LeadJet/p2dRingCutsV0MassPeak"))->GetXaxis()->SetBinLabel(4, "#Lambda + Jet cuts");

    // Same for subleading jet and leading particle:
    histos.add("IntegratedCuts/SubJet/p2dRingCutsV0MassPeak", "p2dRingCutsSubLeadingJet V0MassPeak; ; SidebandWindow (0) or InPeak (1);<#it{R}>", kTProfile2D, {{4, 0, 4}, {2, 0, 2}});
    histos.get<TProfile2D>(HIST("IntegratedCuts/SubJet/p2dRingCutsV0MassPeak"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
    histos.get<TProfile2D>(HIST("IntegratedCuts/SubJet/p2dRingCutsV0MassPeak"))->GetXaxis()->SetBinLabel(2, "p_{T,#Lambda}@[0.5,1.5],|y_{#Lambda}|<0.5");
    histos.get<TProfile2D>(HIST("IntegratedCuts/SubJet/p2dRingCutsV0MassPeak"))->GetXaxis()->SetBinLabel(3, "|SubJet_{#eta}|<0.5");
    histos.get<TProfile2D>(HIST("IntegratedCuts/SubJet/p2dRingCutsV0MassPeak"))->GetXaxis()->SetBinLabel(4, "#Lambda + SubJet cuts");

    histos.add("IntegratedCuts/LeadP/p2dRingCutsV0MassPeak", "p2dRingCutsLeadingP V0MassPeak; ; SidebandWindow (0) or InPeak (1);<#it{R}>", kTProfile2D, {{4, 0, 4}, {2, 0, 2}});
    histos.get<TProfile2D>(HIST("IntegratedCuts/LeadP/p2dRingCutsV0MassPeak"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
    histos.get<TProfile2D>(HIST("IntegratedCuts/LeadP/p2dRingCutsV0MassPeak"))->GetXaxis()->SetBinLabel(2, "p_{T}^{#Lambda}@[0.5,1.5],|y_{#Lambda}|<0.5");
    histos.get<TProfile2D>(HIST("IntegratedCuts/LeadP/p2dRingCutsV0MassPeak"))->GetXaxis()->SetBinLabel(3, "|LeadP_{#eta}|<0.5");
    histos.get<TProfile2D>(HIST("IntegratedCuts/LeadP/p2dRingCutsV0MassPeak"))->GetXaxis()->SetBinLabel(4, "#Lambda + LeadP cuts");

    // Counters for each case to understand statistics loss:
    histos.add("IntegratedCuts/LeadJet/hCountCuts", "hCountCuts; ;N V0s", kTH1D, {{4, 0, 4}});
    histos.get<TH1>(HIST("IntegratedCuts/LeadJet/hCountCuts"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
    histos.get<TH1>(HIST("IntegratedCuts/LeadJet/hCountCuts"))->GetXaxis()->SetBinLabel(2, "p_{T}^{#Lambda}@[0.5,1.5],|y_{#Lambda}|<0.5"); // (v0pt > 0.5 && v0pt < 1.5) && std::abs(lambdaRapidity) < 0.5;
    histos.get<TH1>(HIST("IntegratedCuts/LeadJet/hCountCuts"))->GetXaxis()->SetBinLabel(3, "|Jet_{#eta}|<0.5");
    histos.get<TH1>(HIST("IntegratedCuts/LeadJet/hCountCuts"))->GetXaxis()->SetBinLabel(4, "#Lambda + Jet cuts");

    // Same for subleading jet and leading particle:
    histos.add("IntegratedCuts/SubJet/hCountCuts", "hCountCuts; ;N V0s", kTH1D, {{4, 0, 4}});
    histos.get<TH1>(HIST("IntegratedCuts/SubJet/hCountCuts"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
    histos.get<TH1>(HIST("IntegratedCuts/SubJet/hCountCuts"))->GetXaxis()->SetBinLabel(2, "p_{T,#Lambda}@[0.5,1.5],|y_{#Lambda}|<0.5");
    histos.get<TH1>(HIST("IntegratedCuts/SubJet/hCountCuts"))->GetXaxis()->SetBinLabel(3, "|SubJet_{#eta}|<0.5");
    histos.get<TH1>(HIST("IntegratedCuts/SubJet/hCountCuts"))->GetXaxis()->SetBinLabel(4, "#Lambda + SubJet cuts");

    histos.add("IntegratedCuts/LeadP/hCountCuts", "hCountCuts; ;N V0s", kTH1D, {{4, 0, 4}});
    histos.get<TH1>(HIST("IntegratedCuts/LeadP/hCountCuts"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
    histos.get<TH1>(HIST("IntegratedCuts/LeadP/hCountCuts"))->GetXaxis()->SetBinLabel(2, "p_{T}^{#Lambda}@[0.5,1.5],|y_{#Lambda}|<0.5");
    histos.get<TH1>(HIST("IntegratedCuts/LeadP/hCountCuts"))->GetXaxis()->SetBinLabel(3, "|LeadP_{#eta}|<0.5");
    histos.get<TH1>(HIST("IntegratedCuts/LeadP/hCountCuts"))->GetXaxis()->SetBinLabel(4, "#Lambda + LeadP cuts");

    // Fake-polarization diagnostics:
    if (qaSwitches.doFakePolDiagnosticsQA) {
      // Integrated observable dependent on jet proxy #eta to unfold possible asymmetries in detector:
      histos.add("EtaStudy/LeadJet/pRingEtaCuts", "pRingEtaCuts; ;<#it{R}>", kTProfile, {{15, 0, 15}});
      histos.get<TProfile>(HIST("EtaStudy/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TProfile>(HIST("EtaStudy/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(2, "#eta_{Jet} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(3, "#eta_{Jet} < 0");
      histos.get<TProfile>(HIST("EtaStudy/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("EtaStudy/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(6, "#eta_{Jet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(7, "#eta_{Jet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("EtaStudy/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(8, "#eta_{Jet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(9, "#eta_{Jet} < 0, #eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("EtaStudy/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(10, "#eta_{Jet} > R");
      histos.get<TProfile>(HIST("EtaStudy/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(11, "#eta_{Jet} < -R");
      histos.get<TProfile>(HIST("EtaStudy/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(12, "#eta_{Jet} > R, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(13, "#eta_{Jet} > R, #eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("EtaStudy/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(14, "#eta_{Jet} < -R, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(15, "#eta_{Jet} < -R, #eta_{#Lambda} < 0");

      histos.add("EtaStudy/SubJet/pRingEtaCuts", "pRingEtaCuts; ;<#it{R}>", kTProfile, {{15, 0, 15}});
      histos.get<TProfile>(HIST("EtaStudy/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TProfile>(HIST("EtaStudy/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(2, "#eta_{SubJet} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(3, "#eta_{SubJet} < 0");
      histos.get<TProfile>(HIST("EtaStudy/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("EtaStudy/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(6, "#eta_{SubJet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(7, "#eta_{SubJet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("EtaStudy/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(8, "#eta_{SubJet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(9, "#eta_{SubJet} < 0, #eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("EtaStudy/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(10, "#eta_{SubJet} > R");
      histos.get<TProfile>(HIST("EtaStudy/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(11, "#eta_{SubJet} < -R");
      histos.get<TProfile>(HIST("EtaStudy/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(12, "#eta_{SubJet} > R, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(13, "#eta_{SubJet} > R, #eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("EtaStudy/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(14, "#eta_{SubJet} < -R, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(15, "#eta_{SubJet} < -R, #eta_{#Lambda} < 0");

      histos.add("EtaStudy/LeadP/pRingEtaCuts", "pRingEtaCuts; ;<#it{R}>", kTProfile, {{9, 0, 9}});
      histos.get<TProfile>(HIST("EtaStudy/LeadP/pRingEtaCuts"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TProfile>(HIST("EtaStudy/LeadP/pRingEtaCuts"))->GetXaxis()->SetBinLabel(2, "#eta_{LeadP} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/LeadP/pRingEtaCuts"))->GetXaxis()->SetBinLabel(3, "#eta_{LeadP} < 0");
      histos.get<TProfile>(HIST("EtaStudy/LeadP/pRingEtaCuts"))->GetXaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/LeadP/pRingEtaCuts"))->GetXaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("EtaStudy/LeadP/pRingEtaCuts"))->GetXaxis()->SetBinLabel(6, "#eta_{LeadP} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/LeadP/pRingEtaCuts"))->GetXaxis()->SetBinLabel(7, "#eta_{LeadP} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("EtaStudy/LeadP/pRingEtaCuts"))->GetXaxis()->SetBinLabel(8, "#eta_{LeadP} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/LeadP/pRingEtaCuts"))->GetXaxis()->SetBinLabel(9, "#eta_{LeadP} < 0, #eta_{#Lambda} < 0");

      // Studying the signal Vs background integral (a very naive estimative of the invariant mass peak)
      histos.add("EtaStudy/LeadP/p2dRingEtaCuts_MassSignalVsBackground", "p2dRingEtaCuts_MassSignalVsBackground; ; ;<#it{R}>", kTProfile2D, {{9, 0, 9}, {2, 0, 2}});
      histos.get<TProfile2D>(HIST("EtaStudy/LeadP/p2dRingEtaCuts_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadP/p2dRingEtaCuts_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(2, "#eta_{LeadP} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadP/p2dRingEtaCuts_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(3, "#eta_{LeadP} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadP/p2dRingEtaCuts_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadP/p2dRingEtaCuts_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadP/p2dRingEtaCuts_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(6, "#eta_{LeadP} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadP/p2dRingEtaCuts_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(7, "#eta_{LeadP} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadP/p2dRingEtaCuts_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(8, "#eta_{LeadP} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadP/p2dRingEtaCuts_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(9, "#eta_{LeadP} < 0, #eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadP/p2dRingEtaCuts_MassSignalVsBackground"))->GetYaxis()->SetBinLabel(1, "#Lambda out of mass peak");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadP/p2dRingEtaCuts_MassSignalVsBackground"))->GetYaxis()->SetBinLabel(2, "#Lambda in mass peak");

      // Fake polarization signal QA
      // --> The "negative helicity problem", where topologies with a proton decaying opposite to the Lambda momentum are enhanced by
      // efficiency of reconstruction. The geometries where the proton moves in the same direction as the boost will have a very small
      // momentum pion, which is not as easily detected as the opposite case! This may insert a fake signal of polarization in the measurement!
      histos.add("EtaStudy/LeadJet/h2dFakePolCounts", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda};", kTH2D, {axisConfigurations.axisCosTheta, {9, 0, 9}});
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCounts"))->GetZaxis()->SetTitle("N_{V0s}");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(2, "#eta_{Jet} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(3, "#eta_{Jet} < 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(6, "#eta_{Jet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(7, "#eta_{Jet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(8, "#eta_{Jet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(9, "#eta_{Jet} < 0, #eta_{#Lambda} < 0");

      // The same, but for actual signal instead of counts:
      histos.add("EtaStudy/LeadJet/p2dFakePolSignalVsCosTheta", "FakePolSignal; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda};", kTProfile2D, {axisConfigurations.axisCosTheta, {9, 0, 9}});
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsCosTheta"))->GetZaxis()->SetTitle("<#it{R}>");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(2, "#eta_{Jet} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(3, "#eta_{Jet} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(6, "#eta_{Jet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(7, "#eta_{Jet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(8, "#eta_{Jet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(9, "#eta_{Jet} < 0, #eta_{#Lambda} < 0");

      // Seeing the dependence between phi* = atan2(p_p_star \cdot (p_Lambda_hat \times (z_hat \cross p_Lambda)), p_p_star \cdot (z_hat \cross p_Lambda))
      // e_z = p_Lambda_hat; // e_x = normalize(z_hat cross p_Lambda); // e_y = e_z cross e_x;
      // phi_star = atan2(p_p_star dot e_y, p_p_star dot e_x);
      histos.add("HelicityEfficiencyQA/h2dFakePolCounts_CosThetaVsPhiStar", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #phi^{*}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaPhi});
      histos.add("HelicityEfficiencyQA/LeadJet/p2dFakePolSignal_CosThetaVsPhiStar", "FakePolSignal; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #phi^{*}", kTProfile2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaPhi});
      // Specific counter for when we have leading jets (relates directly to LeadJet/p2dFakePolSignal_CosThetaVsPhiStar):
      histos.add("HelicityEfficiencyQA/LeadJet/h2dFakePolCounts_CosThetaVsPhiStar", "FakePolCounts - HasValidLeadJet OK; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #phi^{*}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaPhi});

      // Similar split, but for AEE instead of HEE:
      histos.add("EtaStudy/LeadJet/h2dCountsVsPhiStar", "FakePolCounts, AEE dependence; #phi^{*};", kTH2D, {axisConfigurations.axisDeltaPhi, {9, 0, 9}});
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dCountsVsPhiStar"))->GetZaxis()->SetTitle("Counts");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dCountsVsPhiStar"))->GetYaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dCountsVsPhiStar"))->GetYaxis()->SetBinLabel(2, "#eta_{Jet} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dCountsVsPhiStar"))->GetYaxis()->SetBinLabel(3, "#eta_{Jet} < 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dCountsVsPhiStar"))->GetYaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dCountsVsPhiStar"))->GetYaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dCountsVsPhiStar"))->GetYaxis()->SetBinLabel(6, "#eta_{Jet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dCountsVsPhiStar"))->GetYaxis()->SetBinLabel(7, "#eta_{Jet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dCountsVsPhiStar"))->GetYaxis()->SetBinLabel(8, "#eta_{Jet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dCountsVsPhiStar"))->GetYaxis()->SetBinLabel(9, "#eta_{Jet} < 0, #eta_{#Lambda} < 0");
      // For the ring observable as well:
      // Explicitly, checking the Phi* dependence on a series of #eta cuts.
      // The fake, AEE-induced, signal should be zero when integrating on full solid angle, and then have some shape for each eta slice.
      histos.add("EtaStudy/LeadJet/p2dFakePolSignalVsPhiStar", "FakePolSignal, AEE dependence; #phi^{*};", kTProfile2D, {axisConfigurations.axisDeltaPhi, {9, 0, 9}});
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiStar"))->GetZaxis()->SetTitle("<#it{R}>");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiStar"))->GetYaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiStar"))->GetYaxis()->SetBinLabel(2, "#eta_{Jet} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiStar"))->GetYaxis()->SetBinLabel(3, "#eta_{Jet} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiStar"))->GetYaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiStar"))->GetYaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiStar"))->GetYaxis()->SetBinLabel(6, "#eta_{Jet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiStar"))->GetYaxis()->SetBinLabel(7, "#eta_{Jet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiStar"))->GetYaxis()->SetBinLabel(8, "#eta_{Jet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiStar"))->GetYaxis()->SetBinLabel(9, "#eta_{Jet} < 0, #eta_{#Lambda} < 0");

      // For the phi_Lambda - phi_D^* dependency as well:
      histos.add("EtaStudy/LeadJet/p2dFakePolSignalVsPhiAEE", "FakePolSignal, AEE dependence; #phi_{#Lambda}-#phi_{p}^{*};", kTProfile2D, {axisConfigurations.axisDeltaPhi, {9, 0, 9}});
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiAEE"))->GetZaxis()->SetTitle("<#it{R}>");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiAEE"))->GetYaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiAEE"))->GetYaxis()->SetBinLabel(2, "#eta_{Jet} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiAEE"))->GetYaxis()->SetBinLabel(3, "#eta_{Jet} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiAEE"))->GetYaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiAEE"))->GetYaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiAEE"))->GetYaxis()->SetBinLabel(6, "#eta_{Jet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiAEE"))->GetYaxis()->SetBinLabel(7, "#eta_{Jet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiAEE"))->GetYaxis()->SetBinLabel(8, "#eta_{Jet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiAEE"))->GetYaxis()->SetBinLabel(9, "#eta_{Jet} < 0, #eta_{#Lambda} < 0");

      // More about possible AEE dependencies (should see an invariance with JetEta and <R>/jetZ):
      // TODO: think about these error bars: do they still make sense via regular TProfile's SEM error? These are just a quick check, so wouldn't bother much about it.
      histos.add("HelicityEfficiencyQA/LeadJet/ProxyZ/pRingVsProxyZ", "<#it{R}> vs #hat{t}_{z}; #hat{t}_{z}; <#it{R}>", kTProfile, {axisConfigurations.axisProxyZ}); // Numerically stable and can also show the sign flip (essentially the <#it{R}> vs Eta Jet plot in another scale)
      histos.add("HelicityEfficiencyQA/LeadJet/ProxyZ/pRingOverProxyZ_VsProxyEta", "<#it{R}>/#hat{t}_{z}; #eta_{Jet}; <#it{R}>/#hat{t}_{z}", kTProfile, {axisConfigurations.axisEtaCoarse});
      histos.add("HelicityEfficiencyQA/LeadJet/ProxyZ/pRingOverProxyZ_VsCosThetaHEE", "<#it{R}>/#hat{t}_{z} Vs cos(#theta) HEE; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; <#it{R}>/#hat{t}_{z}", kTProfile, {axisConfigurations.axisCosTheta});
      histos.add("HelicityEfficiencyQA/LeadJet/ProxyZ/pRingOverProxyZ_VsPhiStar", "<#it{R}>/#hat{t}_{z} Vs #phi^{*}; #phi^{*} = atan2(#vec{p}^{*}_{p} #upoint [#hat{p}_{#Lambda} #times (#hat{z} #times #hat{p}_{#Lambda})] , #vec{p}^{*}_{p} #upoint (#hat{z} #times #hat{p}_{#Lambda})); <#it{R}>/#hat{t}_{z}", kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("HelicityEfficiencyQA/LeadJet/ProxyZ/p2dRingOverProxyZ_VsProxyEtaVsCosThetaHEE", "<#it{R}>/#hat{t}_{z}; #eta_{Jet}; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; <#it{R}>/#hat{t}_{z}", kTProfile2D, {axisConfigurations.axisEtaCoarse, axisConfigurations.axisCosTheta});
      histos.add("HelicityEfficiencyQA/LeadJet/ProxyZ/p2dRingOverProxyZ_VsProxyEtaVsPhiStar", "<#it{R}>/#hat{t}_{z}; #eta_{Jet}; #phi^{*}; <#it{R}>/#hat{t}_{z}", kTProfile2D, {axisConfigurations.axisEtaCoarse, axisConfigurations.axisDeltaPhi});

      // Seeing if phi* is indeed influenced by the DCA between daughters:
      histos.add("HelicityEfficiencyQA/LeadJet/h2dFakePolCounts_PhiStarVsDCAdau", "FakePolCounts - HasValidLeadJet OK; #phi^{*}; DCA_{V0 Daughters}", kTH2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdau});
      histos.add("HelicityEfficiencyQA/LeadJet/p2dFakePolSignal_PhiStarVsDCAdau", "FakePolSignal; #phi^{*}; DCA_{V0 Daughters}", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdau});

      // Adding a way to check if the jet eta is positive or negative as well
      // (AEE signal could be closer to zero otherwise: phi^* dependency may not make it fall to zero as we are no longer integrating <R> in full solid angle, yet analyzing this other dependency is also important)
      histos.add("HelicityEfficiencyQA/LeadJet/p3dFakePolSignal_PhiStarVsDCAdauVsProxyEta", "FakePolSignal; #phi^{*}; DCA_{V0 Daughters}; #eta_{Jet} sign", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdau, {2, -0.9, 0.9}});
      histos.add("HelicityEfficiencyQA/LeadJet/p3dFakePolSignal_PhiStarVsDCAdauVsEtaLambda", "FakePolSignal; #phi^{*}; DCA_{V0 Daughters}; #eta_{#Lambda} sign", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdau, {2, -0.9, 0.9}});

      // Similar checks for dcaPosToPV and dcaNegToPV, which influence AEE the strongest:
      histos.add("HelicityEfficiencyQA/LeadJet/h2dFakePolCounts_PhiStarVsDCAProLike", "FakePolCounts - HasValidLeadJet OK; #phi^{*}; DCA_{PosPV}", kTH2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdauPV});
      histos.add("HelicityEfficiencyQA/LeadJet/p2dFakePolSignal_PhiStarVsDCAProLike", "FakePolSignal; #phi^{*}; DCA_{PosPV}", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdauPV});
      histos.add("HelicityEfficiencyQA/LeadJet/p3dFakePolSignal_PhiStarVsDCAProLikeVsProxyEta", "FakePolSignal; #phi^{*}; DCA_{PosPV}; #eta_{Jet} sign", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdauPV, {2, -0.9, 0.9}});
      histos.add("HelicityEfficiencyQA/LeadJet/p3dFakePolSignal_PhiStarVsDCAProLikeVsEtaLambda", "FakePolSignal; #phi^{*}; DCA_{PosPV}; #eta_{#Lambda} sign", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdauPV, {2, -0.9, 0.9}});

      histos.add("HelicityEfficiencyQA/LeadJet/h2dFakePolCounts_PhiStarVsDCAPiLike", "FakePolCounts - HasValidLeadJet OK; #phi^{*}; DCA_{NegPV}", kTH2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdauPV});
      histos.add("HelicityEfficiencyQA/LeadJet/p2dFakePolSignal_PhiStarVsDCAPiLike", "FakePolSignal; #phi^{*}; DCA_{NegPV}", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdauPV});
      histos.add("HelicityEfficiencyQA/LeadJet/p3dFakePolSignal_PhiStarVsDCAPiLikeVsProxyEta", "FakePolSignal; #phi^{*}; DCA_{NegPV}; #eta_{Jet} sign", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdauPV, {2, -0.9, 0.9}});
      histos.add("HelicityEfficiencyQA/LeadJet/p3dFakePolSignal_PhiStarVsDCAPiLikeVsEtaLambda", "FakePolSignal; #phi^{*}; DCA_{NegPV}; #eta_{#Lambda} sign", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdauPV, {2, -0.9, 0.9}});

      // Doing the same HEE study for leading particles:
      // (eta_{Jet} may be a bad estimator!)
      histos.add("EtaStudy/LeadP/h2dFakePolCounts", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda};", kTH2D, {axisConfigurations.axisCosTheta, {9, 0, 9}});
      histos.get<TH2>(HIST("EtaStudy/LeadP/h2dFakePolCounts"))->GetZaxis()->SetTitle("N_{V0s}");
      histos.get<TH2>(HIST("EtaStudy/LeadP/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TH2>(HIST("EtaStudy/LeadP/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(2, "#eta_{LeadP} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/LeadP/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(3, "#eta_{LeadP} < 0");
      histos.get<TH2>(HIST("EtaStudy/LeadP/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/LeadP/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("EtaStudy/LeadP/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(6, "#eta_{LeadP} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/LeadP/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(7, "#eta_{LeadP} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("EtaStudy/LeadP/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(8, "#eta_{LeadP} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/LeadP/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(9, "#eta_{LeadP} < 0, #eta_{#Lambda} < 0");

      // Avoid fake signal by jets boosting the Lambda in its own direction, then modifying efficiency of reconstruction in a similar way:
      histos.add("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtCut", "FakePol,p_{T}^{#Lambda}#in[0.5,1.5]; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda};", kTH2D, {axisConfigurations.axisCosTheta, {9, 0, 9}});
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtCut"))->GetZaxis()->SetTitle("N_{V0s}");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(2, "#eta_{Jet} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(3, "#eta_{Jet} < 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(6, "#eta_{Jet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(7, "#eta_{Jet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(8, "#eta_{Jet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(9, "#eta_{Jet} < 0, #eta_{#Lambda} < 0");

      // Even stricter cut (also demands rapidity cut stricter than jets, so may see different boosting):
      histos.add("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtYCuts", "FakePol,p_{T}^{#Lambda}#in[0.5,1.5],|y_{#Lambda}|<0.5; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda};", kTH2D, {axisConfigurations.axisCosTheta, {9, 0, 9}});
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtYCuts"))->GetZaxis()->SetTitle("N_{V0s}");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(2, "#eta_{Jet} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(3, "#eta_{Jet} < 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(6, "#eta_{Jet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(7, "#eta_{Jet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(8, "#eta_{Jet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(9, "#eta_{Jet} < 0, #eta_{#Lambda} < 0");

      // Another useful quantity -- How much is the fake signal related to the jet's momentum (how much the fake signal is correlated with the Jet-Lambda angular separation):
      histos.add("HelicityEfficiencyQA/LeadJet/h2dFakePolCountsVsDeltaTheta", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #Delta#theta_{Jet}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaTheta});
      histos.add("HelicityEfficiencyQA/LeadJet/h2dFakePolCountsVsDeltaThetaPosProxyEta", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #Delta#theta_{Jet}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaTheta});
      histos.add("HelicityEfficiencyQA/LeadJet/h2dFakePolCountsVsDeltaThetaNegProxyEta", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #Delta#theta_{Jet}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaTheta});
      histos.add("HelicityEfficiencyQA/LeadP/h2dFakePolCountsVsDeltaTheta", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #Delta#theta_{LeadP}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaTheta});
      histos.add("HelicityEfficiencyQA/LeadP/h2dFakePolCountsVsDeltaThetaPosProxyEta", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #Delta#theta_{LeadP}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaTheta});
      histos.add("HelicityEfficiencyQA/LeadP/h2dFakePolCountsVsDeltaThetaNegProxyEta", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #Delta#theta_{LeadP}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaTheta});

      // Understanding the dip at the cos = -1 end:
      histos.add("HelicityEfficiencyQA/LeadJet/h2dFakePolCountsCosThetaVsLambdaPt", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; p_{T}^{#Lambda}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisPtCoarseQA});
      histos.add("HelicityEfficiencyQA/LeadP/h2dFakePolCountsCosThetaVsLambdaPt", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; p_{T}^{#Lambda}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisPtCoarseQA});

      // Studying the magnetic field dependence of particle reconstruction efficiency (not magnitude, just sign of field):
      // (also for the "negative helicity" problem)
      if (analyseMagField) {
        if (analyseLambda) { // Fills themselves are guarded at the start of the "v0sInColl" loop, for each of these gated histogram bookings
          histos.add("HelicityEfficiencyQA/hLambdaMassDecayGeomRight", "hLambdaMassDecayGeomRight; m_{Inv}; Counts", kTH1D, {axisConfigurations.axisLambdaMass});
          histos.add("HelicityEfficiencyQA/hLambdaMassDecayGeomLeft", "hLambdaMassDecayGeomLeft; m_{Inv}; Counts", kTH1D, {axisConfigurations.axisLambdaMass});
        }
        if (analyseAntiLambda) {
          histos.add("HelicityEfficiencyQA/hAntiLambdaMassDecayGeomRight", "hAntiLambdaMassDecayGeomRight; m_{Inv}; Counts", kTH1D, {axisConfigurations.axisLambdaMass});
          histos.add("HelicityEfficiencyQA/hAntiLambdaMassDecayGeomLeft", "hAntiLambdaMassDecayGeomLeft; m_{Inv}; Counts", kTH1D, {axisConfigurations.axisLambdaMass});
        }
      }

      // Also including an observable that probes spectrum broadening due to the "Azimuthal Efficiency Effect":
       if (analyseLambda)
        histos.add("HelicityEfficiencyQA/PhiAEE/h2dLambdaMassVsPhiAEE", "m_{#Lambda}, AEE probe; m_{Inv}; #phi_{#Lambda}-#phi_{p}^{*} ; Counts", kTH2D, {axisConfigurations.axisLambdaMass, axisConfigurations.axisDeltaPhi});
      if (analyseAntiLambda)
        histos.add("HelicityEfficiencyQA/PhiAEE/h2dAntiLambdaMassVsPhiAEE", "m_{#bar{#Lambda}}, AEE probe; m_{Inv}; #phi_{#bar{#Lambda}} - #phi_{p}^{*} ; Counts", kTH2D, {axisConfigurations.axisLambdaMass, axisConfigurations.axisDeltaPhi});

      // The Helicity Efficiency Effect resolved in three mass slices (left sideband, peak, right sideband).
      histos.add("HelicityEfficiencyQA/CosThetaHEEByMass/h2dCountsCosThetaHEEVsFullMass", "Counts vs cos(#theta)_{HEE} vs mass, all V0s;cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda-like};m_{p#pi} (GeV/c^{2});<#it{R}>", kTH2D, {axisConfigurations.axisCosThetaCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingCosThetaHEEVsFullMass", "<#it{R}>_{LeadJet} vs cos(#theta)_{HEE} vs mass slice, all V0s;cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda-like};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisCosThetaCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingCosThetaHEEVsMass", "<#it{R}>_{LeadJet} vs cos(#theta)_{HEE} vs mass slice, all V0s;cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda-like};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisCosThetaCoarse, axisConfigurations.axisLambdaMassThreeBin});
      histos.add("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingCosThetaHEEVsMassPosPhiAEE", "<#it{R}>_{LeadJet} (#phi_{#Lambda}-#phi_{p}^{*}>0) vs cos(#theta)_{HEE} vs mass slice, all V0s;cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda-like};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisCosThetaCoarse, axisConfigurations.axisLambdaMassThreeBin});
      histos.add("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingCosThetaHEEVsMassNegPhiAEE", "<#it{R}>_{LeadJet} (#phi_{#Lambda}-#phi_{p}^{*}<0) vs cos(#theta)_{HEE} vs mass slice, all V0s;cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda-like};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisCosThetaCoarse, axisConfigurations.axisLambdaMassThreeBin});

      // <R> Vs PhiLambda-PhiProtonStar dependencies:
      // (species-agnostic: the population is set by analyseLambda/analyseAntiLambda, so the species split lives in separate configs)
      histos.add("HelicityEfficiencyQA/PhiAEE/LeadJet/pRingObservableVsPhiAEE", "<#it{R}>_{LeadJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*};#phi_{#Lambda-like}-#phi_{p-like}^{*};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("HelicityEfficiencyQA/PhiAEE/LeadP/pRingObservableVsPhiAEE", "<#it{R}>_{LeadP} vs #phi_{#Lambda-like}-#phi_{p-like}^{*};#phi_{#Lambda-like}-#phi_{p-like}^{*};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("HelicityEfficiencyQA/PhiAEE/SubJet/pRingObservableVsPhiAEE", "<#it{R}>_{SubJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*};#phi_{#Lambda-like}-#phi_{p-like}^{*};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaPhi});

      // 2D dependencies for signal extraction:
      // (coarser angular axis here: the mass axis takes the statistics, and each cell still has to sustain a fit)
      histos.add("HelicityEfficiencyQA/PhiAEE/LeadJet/p2dRingObservableVsPhiAEEVsMass", "<#it{R}>_{LeadJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiAEE/LeadP/p2dRingObservableVsPhiAEEVsMass", "<#it{R}>_{LeadP} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiAEE/SubJet/p2dRingObservableVsPhiAEEVsMass", "<#it{R}>_{SubJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      // Splitting in proxy eta:
      histos.add("HelicityEfficiencyQA/PhiAEE/LeadJet/p2dRingObservableVsPhiAEEVsMassPosProxyEta", "<#it{R}>_{LeadJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass, #eta_{Proxy}>0;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiAEE/LeadP/p2dRingObservableVsPhiAEEVsMassPosProxyEta", "<#it{R}>_{LeadP} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass, #eta_{Proxy}>0;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiAEE/SubJet/p2dRingObservableVsPhiAEEVsMassPosProxyEta", "<#it{R}>_{SubJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass, #eta_{Proxy}>0;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiAEE/LeadJet/p2dRingObservableVsPhiAEEVsMassNegProxyEta", "<#it{R}>_{LeadJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass, #eta_{Proxy}<0;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiAEE/LeadP/p2dRingObservableVsPhiAEEVsMassNegProxyEta", "<#it{R}>_{LeadP} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass, #eta_{Proxy}<0;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiAEE/SubJet/p2dRingObservableVsPhiAEEVsMassNegProxyEta", "<#it{R}>_{SubJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass, #eta_{Proxy}<0;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      // AEE angle Vs Proxy eta dependence:
      histos.add("HelicityEfficiencyQA/PhiAEE/LeadJet/p2dRingObservableVsPhiAEEVsProxyEtaSign", "<#it{R}>_{LeadJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy} sign;<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, {2, -1, 1}});
      histos.add("HelicityEfficiencyQA/PhiAEE/LeadP/p2dRingObservableVsPhiAEEVsProxyEtaSign", "<#it{R}>_{LeadP} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy} sign;<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, {2, -1, 1}});
      histos.add("HelicityEfficiencyQA/PhiAEE/SubJet/p2dRingObservableVsPhiAEEVsProxyEtaSign", "<#it{R}>_{SubJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy} sign;<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, {2, -1, 1}});
      histos.add("HelicityEfficiencyQA/PhiAEE/LeadJet/p2dRingObservableVsPhiAEEVsProxyEta", "<#it{R}>_{LeadJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy};<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEta});
      histos.add("HelicityEfficiencyQA/PhiAEE/LeadP/p2dRingObservableVsPhiAEEVsProxyEta", "<#it{R}>_{LeadP} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy};<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEta});
      histos.add("HelicityEfficiencyQA/PhiAEE/SubJet/p2dRingObservableVsPhiAEEVsProxyEta", "<#it{R}>_{SubJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy};<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEta});
      // A 3D split that allows for a proper signal extraction after separating the main drivers of fake polarization signal:
      histos.add("HelicityEfficiencyQA/PhiAEE/LeadJet/p3dRingObservableVsPhiAEEVsProxyEtaVsMass", "<#it{R}>_{LeadJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy}; m_{p#pi};<#it{R}>", kTProfile3D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiAEE/LeadP/p3dRingObservableVsPhiAEEVsProxyEtaVsMass", "<#it{R}>_{LeadP} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy}; m_{p#pi};<#it{R}>", kTProfile3D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiAEE/SubJet/p3dRingObservableVsPhiAEEVsProxyEtaVsMass", "<#it{R}>_{SubJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy}; m_{p#pi};<#it{R}>", kTProfile3D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});

      // The same profiles collapsed onto three mass slices:
      // (high statistics for a simpler lookup)
      histos.add("HelicityEfficiencyQA/PhiAEE/ThreeBinMass/LeadJet/p2dRingObservableVsPhiAEEVs3BinMass", "<#it{R}>_{LeadJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs mass slice;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassThreeBin});
      histos.add("HelicityEfficiencyQA/PhiAEE/ThreeBinMass/LeadP/p2dRingObservableVsPhiAEEVs3BinMass", "<#it{R}>_{LeadP} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs mass slice;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassThreeBin});
      histos.add("HelicityEfficiencyQA/PhiAEE/ThreeBinMass/SubJet/p2dRingObservableVsPhiAEEVs3BinMass", "<#it{R}>_{SubJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs mass slice;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassThreeBin});

      // Picking a representative split in eta and PhiAEE just to see if we manage to stabilize the background (based off p2dRingObservableEtaLambdaVsMass, but booked outside the addRingObservableFamily lambda):
      histos.add("HelicityEfficiencyQA/PhiAEE/BkgStabilizationAttempt/p2dRingObservableEtaLambdaVsMassEtaProxyPosPhiAEEPos", "<#it{R}> vs #eta_{#Lambda} vs Mass, #eta_{Proxy}>0, #phi_{AEE}>0;#eta_{#Lambda};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiAEE/BkgStabilizationAttempt/p2dRingObservableEtaLambdaVsMassEtaProxyNegPhiAEEPos", "<#it{R}> vs #eta_{#Lambda} vs Mass, #eta_{Proxy}<0, #phi_{AEE}>0;#eta_{#Lambda};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiAEE/BkgStabilizationAttempt/p2dRingObservableEtaLambdaVsMassEtaProxyPosPhiAEEPosDeltaPhiLeadJetPos", "<#it{R}> vs #eta_{#Lambda} vs Mass, #eta_{Proxy}>0, #phi_{AEE}>0, #Delta#phi_{Jet}>0;#eta_{#Lambda};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiAEE/BkgStabilizationAttempt/p2dRingObservableEtaLambdaVsMassEtaProxyPosPhiAEEPosDeltaPhiLeadJetNeg", "<#it{R}> vs #eta_{#Lambda} vs Mass, #eta_{Proxy}<0, #phi_{AEE}>0, #Delta#phi_{Jet}<0;#eta_{#Lambda};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisLambdaMassSigExtract});
      // Splitting in the jet coordinate, but leaving PhiAEE differential:
      histos.add("HelicityEfficiencyQA/PhiAEE/BkgStabilizationAttempt/p2dRingObservablePhiAEEVsMassEtaProxyPosPhiEtaLambdaPos", "<#it{R}> vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} vs Mass, #eta_{Proxy}>0, #eta_{#Lambda}>0;#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiAEE/BkgStabilizationAttempt/p2dRingObservablePhiAEEVsMassEtaProxyNegPhiEtaLambdaPos", "<#it{R}> vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} vs Mass, #eta_{Proxy}<0, #eta_{#Lambda}>0;#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiAEE/BkgStabilizationAttempt/p2dRingObservablePhiAEEVsMassEtaProxyPosPhiEtaLambdaPosDeltaPhiLeadJetPos", "<#it{R}> vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} vs Mass, #eta_{Proxy}>0, #eta_{#Lambda}>0, #Delta#phi_{Jet}>0;#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiAEE/BkgStabilizationAttempt/p2dRingObservablePhiAEEVsMassEtaProxyPosPhiEtaLambdaPosDeltaPhiLeadJetNeg", "<#it{R}> vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} vs Mass, #eta_{Proxy}<0, #eta_{#Lambda}>0, #Delta#phi_{Jet}<0;#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
    } // end doFakePolDiagnosticsQA bookings

    // Integrated observable for events with NLambda+NAntiLambda V0s per event
    // (an interesting measurement of correlation between <R> and Lambda-like V0s multiplicity. A proxy of covariance)
    // (calculated for leading jets only)
    histos.add("IntegratedCuts/LeadJet/pRingVsNV0s", "pRingVsNV0s; N_{#Lambda}+N_{#bar{#Lambda}};<#it{R}>", kTProfile, {{20, 0, 20}}); // See h2dNV0sVsCentrality below for the correlation between number of V0s and centrality
    histos.add("h2dNV0sVsCentrality", "h2dNV0sVsCentrality; N_{#Lambda}+N_{#bar{#Lambda}};Centrality (%)", kTH2D, {{20, 0, 20}, axisConfigurations.axisCentrality});

    // V0 consumer selection level QAing and signal extraction:
    // (Mirrors GeneralQA/hSelectionV0s from the lambdaJetPolarizationIons.cxx macro, but for isV0Accepted)
    if (analysisLevelCuts.doAnalysisLevelCuts) { // Booked only when the cuts are actually applied
      struct CutLabel {
        std::string label;
        bool enabled; // Same trick as in the TableProducer: selections that are not in use are greyed out
      };
      const std::vector<CutLabel> v0AnalysisCutLabels = {
        {"All consumer V0s", true},
        {"p_{T} (min)", analysisLevelCuts.v0MinPt > 0.f},
        {"p_{T} (max)", analysisLevelCuts.v0MaxPt < 999.f},
        {"|y_{#Lambda}|", analysisLevelCuts.v0MaxRap < 999.f},
        {"|#eta_{#Lambda}|", analysisLevelCuts.v0MaxEta < 999.f},
        {"AP |#alpha| (min)", analysisLevelCuts.apAlphaMin > 0.f},
        {"AP |#alpha| (max)", analysisLevelCuts.apAlphaMax < 1.f},
        {"AP q_{T} (min)", analysisLevelCuts.apQtMin > 0.f},
        {"AP q_{T} (max)", analysisLevelCuts.apQtMax < 999.f},
        {"TPC n#sigma (p-like)", analysisLevelCuts.nSigmaTPCPrLike < 999.f},
        {"TPC n#sigma (#pi-like)", analysisLevelCuts.nSigmaTPCPiLike < 999.f},
        {"DCA_{V0 daughters}", analysisLevelCuts.v0MaxDcaDau < 999.f},
        {"V0 cosPA", analysisLevelCuts.v0MinCosPA > -1.f},
        {"V0 radius (min)", analysisLevelCuts.v0MinRadius > 0.f},
        {"V0 radius (max)", analysisLevelCuts.v0MaxRadius < 999.f},
        {"DCA_{p-like} to PV", analysisLevelCuts.v0MinDcaPrLikeToPV > 0.f},
        {"DCA_{#pi-like} to PV", analysisLevelCuts.v0MinDcaPiLikeToPV > 0.f},
        {"#phi_{V0}", (v0PhiLimitsVec[0] > 0.f || v0PhiLimitsVec[1] < constants::math::TwoPI)},
        {"V0 Mass (min)", analysisLevelCuts.v0MassMin > 0.f},
        {"V0 Mass (max)", analysisLevelCuts.v0MassMax < 999.f},
        {"Final accepted", true},
      };
      const int nAnalysisCutBins = static_cast<int>(v0AnalysisCutLabels.size());
 
      auto hAnalysisLevelSelectionV0s = histos.add<TH1>("hAnalysisLevelSelectionV0s", "Analysis-level V0 selection flow", kTH1D, {{nAnalysisCutBins, -0.5, static_cast<double>(nAnalysisCutBins) - 0.5}});
      // Same flow against the Lambda-like mass, to tell background rejection apart from plain V0 candidate loss (same GeneralQA/h2dSelectionLambdaMass):
      auto h2dAnalysisLevelSelectionV0sVsMass = histos.add<TH2>("h2dAnalysisLevelSelectionV0sVsMass", "Analysis-level V0 selection flow vs M_{#Lambda-like}; ;m_{p#pi} (GeV/c^{2})", kTH2D, {{nAnalysisCutBins, -0.5, static_cast<double>(nAnalysisCutBins) - 0.5}, axisConfigurations.axisLambdaMassSigExtract});
      for (int i = 0; i < nAnalysisCutBins; ++i) {
        auto lbl = v0AnalysisCutLabels[i].label;
        if (!v0AnalysisCutLabels[i].enabled)
          lbl = "#color[16]{(off) " + lbl + "}";
        hAnalysisLevelSelectionV0s->GetXaxis()->SetBinLabel(i + 1, lbl.c_str()); // First non-underflow bin is bin 1
        h2dAnalysisLevelSelectionV0sVsMass->GetXaxis()->SetBinLabel(i + 1, lbl.c_str());
      }
      histos.add("h2dArmenterosInput", "h2dArmenterosInput;Armenteros #alpha;Armenteros q_{T} (GeV/c)", kTH2D, {axisConfigurations.axisAPAlpha, axisConfigurations.axisAPQt});
      histos.add("h2dArmenterosAnalysisCuts", "h2dArmenterosAnalysisCuts;Armenteros #alpha;Armenteros q_{T} (GeV/c)", kTH2D, {axisConfigurations.axisAPAlpha, axisConfigurations.axisAPQt});
    }

    // Proxy Eta QA:
    histos.add("KinematicsQA/LeadJet/hEta", "hEta;#eta;Counts", kTH1D, {axisConfigurations.axisEta});
    histos.add("KinematicsQA/SubJet/hEta", "hEta;#eta;Counts", kTH1D, {axisConfigurations.axisEta});
    histos.add("KinematicsQA/LeadP/hEta", "hEta;#eta;Counts", kTH1D, {axisConfigurations.axisEta});

    // Proxy Phi QA:
    histos.add("KinematicsQA/LeadJet/hPhi", "hPhi;#varphi;Counts", kTH1D, {axisConfigurations.axisPhi});
    histos.add("KinematicsQA/SubJet/hPhi", "hPhi;#varphi;Counts", kTH1D, {axisConfigurations.axisPhi});
    histos.add("KinematicsQA/LeadP/hPhi", "hPhi;#varphi;Counts", kTH1D, {axisConfigurations.axisPhi});

    // Counting the number of jets/proxies themselves (these count at most once per event) -- Similar to the FOLDER/<Proxy>/QA/hProxyPt counters:
    histos.add("KinematicsQA/LeadJet/hPt", "hPt; p_{T}^{Jet} (GeV/c)", kTH1D, {axisConfigurations.axisJetPt});
    histos.add("KinematicsQA/LeadP/hPt", "hPt; p_{T}^{LeadP} (GeV/c)", kTH1D, {axisConfigurations.axisJetPt});
    histos.add("KinematicsQA/SubJet/hPt", "hPt; p_{T}^{SubJet} (GeV/c)", kTH1D, {axisConfigurations.axisJetPt});

    // Proxy Eta vs Proxy Phi:
    histos.add("KinematicsQA/LeadJet/h2dEtaVsPhi", "Lead Jet #eta Vs #phi;#eta;#phi_{Proxy} [rad];Counts", kTH2D, {axisConfigurations.axisEta, axisConfigurations.axisPhi});
    histos.add("KinematicsQA/SubJet/h2dEtaVsPhi", "SubLead Jet #eta Vs #phi;#eta;#phi_{Proxy} [rad];Counts", kTH2D, {axisConfigurations.axisEta, axisConfigurations.axisPhi});
    histos.add("KinematicsQA/LeadP/h2dEtaVsPhi", "Lead Ptc #eta Vs #phi;#eta;#phi_{Proxy} [rad];Counts", kTH2D, {axisConfigurations.axisEta, axisConfigurations.axisPhi});


    // Proxy Eta vs PVz:
    histos.add("KinematicsQA/LeadJet/h2dEtaVsPVz", "Lead Jet #eta Vs PVz;#eta;Primary Vertex Z [cm];Counts", kTH2D, {axisConfigurations.axisEta, axisConfigurations.axisPVz});
    histos.add("KinematicsQA/SubJet/h2dEtaVsPVz", "SubLead Jet #eta Vs PVz;#eta;Primary Vertex Z [cm];Counts", kTH2D, {axisConfigurations.axisEta, axisConfigurations.axisPVz});
    histos.add("KinematicsQA/LeadP/h2dEtaVsPVz", "Lead Ptc #eta Vs PVz;#eta;Primary Vertex Z [cm];Counts", kTH2D, {axisConfigurations.axisEta, axisConfigurations.axisPVz});

    // doMixedEventProxies QA: gauge the size of the "too few collisions per bin" problem (see resonanceMergeDF.cxx)
    // Booked per proxy, since the three mixings are independent and can succeed/fail at different rates.
    if (fakePolSwitches.doMixedEventProxies && qaSwitches.doEventMixingQA) {
      // These histograms are filled in the collision loop, so we are QAing only the collisions with V0s
      histos.add("EventMixingQA/CollLoopOutcome/LeadP/hMixedEventOutcome", "hMixedEventOutcome;Outcome (0=skipped, 1=found);Counts", kTH1D, {{2, -0.5f, 1.5f}});
      histos.add("EventMixingQA/CollLoopOutcome/LeadJet/hMixedEventOutcome", "hMixedEventOutcome;Outcome (0=skipped, 1=found);Counts", kTH1D, {{2, -0.5f, 1.5f}});
      histos.add("EventMixingQA/CollLoopOutcome/SubJet/hMixedEventOutcome", "hMixedEventOutcome;Outcome (0=skipped, 1=found);Counts", kTH1D, {{2, -0.5f, 1.5f}});
        // 2D histograms on every mixing variable:
      histos.add("EventMixingQA/CollLoopOutcome/LeadP/h2dMixedEventOutcomeVsTargetZVtx", "h2dMixedEventOutcomeVsTargetZVtx;Outcome (0=skipped, 1=found);Target primary Vertex Z [cm];Counts", kTH2D, {{2, -0.5f, 1.5f}, axisConfigurations.axisPVz});
      histos.add("EventMixingQA/CollLoopOutcome/LeadJet/h2dMixedEventOutcomeVsTargetZVtx", "h2dMixedEventOutcomeVsTargetZVtx;Outcome (0=skipped, 1=found);Target primary Vertex Z [cm];Counts", kTH2D, {{2, -0.5f, 1.5f}, axisConfigurations.axisPVz});
      histos.add("EventMixingQA/CollLoopOutcome/SubJet/h2dMixedEventOutcomeVsTargetZVtx", "h2dMixedEventOutcomeVsTargetZVtx;Outcome (0=skipped, 1=found);Target primary Vertex Z [cm];Counts", kTH2D, {{2, -0.5f, 1.5f}, axisConfigurations.axisPVz});

      histos.add("EventMixingQA/CollLoopOutcome/LeadP/h2dMixedEventOutcomeVsTargetCentrality", "h2dMixedEventOutcomeVsTargetCentrality;Outcome (0=skipped, 1=found);Target centrality (%);Counts", kTH2D, {{2, -0.5f, 1.5f}, axisConfigurations.axisCentrality});
      histos.add("EventMixingQA/CollLoopOutcome/LeadJet/h2dMixedEventOutcomeVsTargetCentrality", "h2dMixedEventOutcomeVsTargetCentrality;Outcome (0=skipped, 1=found);Target centrality (%);Counts", kTH2D, {{2, -0.5f, 1.5f}, axisConfigurations.axisCentrality});
      histos.add("EventMixingQA/CollLoopOutcome/SubJet/h2dMixedEventOutcomeVsTargetCentrality", "h2dMixedEventOutcomeVsTargetCentrality;Outcome (0=skipped, 1=found);Target centrality (%);Counts", kTH2D, {{2, -0.5f, 1.5f}, axisConfigurations.axisCentrality});

      histos.add("EventMixingQA/CollLoopOutcome/LeadP/h2dMixedEventOutcomeVsTargetProxyPt", "h2dMixedEventOutcomeVsTargetProxyPt;Outcome (0=skipped, 1=found);Target proxy p_{t} [GeV/c];Counts", kTH2D, {{2, -0.5f, 1.5f}, axisConfigurations.axisJetPt});
      histos.add("EventMixingQA/CollLoopOutcome/LeadJet/h2dMixedEventOutcomeVsTargetProxyPt", "h2dMixedEventOutcomeVsTargetProxyPt;Outcome (0=skipped, 1=found);Target proxy p_{t} [GeV/c];Counts", kTH2D, {{2, -0.5f, 1.5f}, axisConfigurations.axisJetPt});
      histos.add("EventMixingQA/CollLoopOutcome/SubJet/h2dMixedEventOutcomeVsTargetProxyPt", "h2dMixedEventOutcomeVsTargetProxyPt;Outcome (0=skipped, 1=found);Target proxy p_{t} [GeV/c];Counts", kTH2D, {{2, -0.5f, 1.5f}, axisConfigurations.axisJetPt});

      // QA at LUT building time:
      histos.add("EventMixingQA/LeadP/hMixedEventWindowNeighbours", "hMixedEventWindowNeighbours;Neighbours found in bin window;Counts", kTH1D, {axisConfigurations.axisMixCandidates});
      histos.add("EventMixingQA/LeadJet/hMixedEventWindowNeighbours", "hMixedEventWindowNeighbours;Neighbours found in bin window;Counts", kTH1D, {axisConfigurations.axisMixCandidates});
      histos.add("EventMixingQA/SubJet/hMixedEventWindowNeighbours", "hMixedEventWindowNeighbours;Neighbours found in bin window;Counts", kTH1D, {axisConfigurations.axisMixCandidates});

      histos.add("EventMixingQA/AnglrCorrltns/LeadP/h2dMixedProxyEtaVsProxyEta", "MixedLeadP #eta vs LeadP #eta;MixedLeadP #eta; LeadP #eta;Counts", kTH2D, {axisConfigurations.axisEtaCoarse, axisConfigurations.axisEtaCoarse});
      histos.add("EventMixingQA/AnglrCorrltns/LeadJet/h2dMixedProxyEtaVsProxyEta", "MixedLeadJet #eta vs LeadJet #eta;MixedLeadJet #eta; LeadJet #eta;Counts", kTH2D, {axisConfigurations.axisEtaCoarse, axisConfigurations.axisEtaCoarse});
      histos.add("EventMixingQA/AnglrCorrltns/SubJet/h2dMixedProxyEtaVsProxyEta", "MixedSubJet #eta vs SubJet #eta;MixedSubJet #eta; SubJet #eta;Counts", kTH2D, {axisConfigurations.axisEtaCoarse, axisConfigurations.axisEtaCoarse});
      histos.add("EventMixingQA/AnglrCorrltns/LeadP/h2dMixedProxyPhiVsProxyPhi", "MixedLeadP #phi vs LeadP #phi;MixedLeadP #phi; LeadP #phi;Counts", kTH2D, {axisConfigurations.axisPhi, axisConfigurations.axisPhi});
      histos.add("EventMixingQA/AnglrCorrltns/LeadJet/h2dMixedProxyPhiVsProxyPhi", "MixedLeadJet #phi vs LeadJet #phi;MixedLeadJet #phi; LeadJet #phi;Counts", kTH2D, {axisConfigurations.axisPhi, axisConfigurations.axisPhi});
      histos.add("EventMixingQA/AnglrCorrltns/SubJet/h2dMixedProxyPhiVsProxyPhi", "MixedSubJet #phi vs SubJet #phi;MixedSubJet #phi; SubJet #phi;Counts", kTH2D, {axisConfigurations.axisPhi, axisConfigurations.axisPhi});

      // Collision-index proximity QA:
      // (To understand possible continous readout effects on the choice of event mixing sources -- do notice index and time are both monotonic, but there is a conversion factor)
      // The shape of the whole candidate pool:
      histos.add("EventMixingQA/IndexQA/LeadP/hMixedEventDeltaIndexEligible", "hMixedEventDeltaIndexEligible;#Delta collision index (pair);Counts", kTH1D, {axisConfigurations.axisDeltaCollisionIndex});
      histos.add("EventMixingQA/IndexQA/LeadJet/hMixedEventDeltaIndexEligible", "hMixedEventDeltaIndexEligible;#Delta collision index (pair);Counts", kTH1D, {axisConfigurations.axisDeltaCollisionIndex});
      histos.add("EventMixingQA/IndexQA/SubJet/hMixedEventDeltaIndexEligible", "hMixedEventDeltaIndexEligible;#Delta collision index (pair);Counts", kTH1D, {axisConfigurations.axisDeltaCollisionIndex});

      // What the reservoir actually picked:
      // (should be equally as narrow as the whole candidate pool, if no preferential mixing exists)
      // This number may be higher than IndexEligible: Eligible has one entry per accepted pair, while Selected has one entry per distinct collision that has at least one mixing candidate.
      histos.add("EventMixingQA/IndexQA/LeadP/hMixedEventDeltaIndexSelected", "hMixedEventDeltaIndexSelected;#Delta collision index (selected partner);Counts", kTH1D, {axisConfigurations.axisDeltaCollisionIndexNonAbs});
      histos.add("EventMixingQA/IndexQA/LeadJet/hMixedEventDeltaIndexSelected", "hMixedEventDeltaIndexSelected;#Delta collision index (selected partner);Counts", kTH1D, {axisConfigurations.axisDeltaCollisionIndexNonAbs});
      histos.add("EventMixingQA/IndexQA/SubJet/hMixedEventDeltaIndexSelected", "hMixedEventDeltaIndexSelected;#Delta collision index (selected partner);Counts", kTH1D, {axisConfigurations.axisDeltaCollisionIndexNonAbs});

      // Event mixing source QA -- How repeated is each collision in the mix:
      // For each source collision for mixing, counts how many times it was used. Flat distribution is ideal.
      histos.add("EventMixingQA/SourceUsage/LeadP/hMixedEventSourceUsageCount", "hMixedEventSourceUsageCount;Times collision was used;Counts", kTH1D, {{50, -0.5f, 49.5f}});
      histos.add("EventMixingQA/SourceUsage/LeadJet/hMixedEventSourceUsageCount", "hMixedEventSourceUsageCount;Times collision was used;Counts", kTH1D, {{50, -0.5f, 49.5f}});
      histos.add("EventMixingQA/SourceUsage/SubJet/hMixedEventSourceUsageCount", "hMixedEventSourceUsageCount;Times collision was used;Counts", kTH1D, {{50, -0.5f, 49.5f}});
      // Useful TProfiles -- the mean number of times a given source was used, as a function of eta or phi
      // (more convenient than a single TH1, as it gives an average, not the raw counter)
      histos.add("EventMixingQA/SourceUsage/LeadP/pMixedEventSourceUsageVsEta", "pMixedEventSourceUsageVsEta;Source #eta;<Times used>", kTProfile, {axisConfigurations.axisEtaCoarse});
      histos.add("EventMixingQA/SourceUsage/LeadP/pMixedEventSourceUsageVsPhi", "pMixedEventSourceUsageVsPhi;Source #varphi;<Times used>", kTProfile, {axisConfigurations.axisPhi});
      histos.add("EventMixingQA/SourceUsage/LeadJet/pMixedEventSourceUsageVsEta", "pMixedEventSourceUsageVsEta;Source #eta;<Times used>", kTProfile, {axisConfigurations.axisEtaCoarse});
      histos.add("EventMixingQA/SourceUsage/LeadJet/pMixedEventSourceUsageVsPhi", "pMixedEventSourceUsageVsPhi;Source #varphi;<Times used>", kTProfile, {axisConfigurations.axisPhi});
      histos.add("EventMixingQA/SourceUsage/SubJet/pMixedEventSourceUsageVsEta", "pMixedEventSourceUsageVsEta;Source #eta;<Times used>", kTProfile, {axisConfigurations.axisEtaCoarse});
      histos.add("EventMixingQA/SourceUsage/SubJet/pMixedEventSourceUsageVsPhi", "pMixedEventSourceUsageVsPhi;Source #varphi;<Times used>", kTProfile, {axisConfigurations.axisPhi});

      // Source-vs-target kinematics for the borrowed proxy, filled before applyProxyDistortion runs:
      // These include "V0less" collisions that can be mixing sources, so it probes a different subset of events from the <Proxy>/hMixedEventOutcome ones.
      histos.add("EventMixingQA/SourceTargetDeltas/LeadP/hMixedEventDeltaPt", "hMixedEventDeltaPt;#Delta p_{T};Counts", kTH1D, {axisConfigurations.axisMixDeltaPt});
      histos.add("EventMixingQA/SourceTargetDeltas/LeadJet/hMixedEventDeltaPt", "hMixedEventDeltaPt;#Delta p_{T};Counts", kTH1D, {axisConfigurations.axisMixDeltaPt});
      histos.add("EventMixingQA/SourceTargetDeltas/SubJet/hMixedEventDeltaPt", "hMixedEventDeltaPt;#Delta p_{T};Counts", kTH1D, {axisConfigurations.axisMixDeltaPt});

      histos.add("EventMixingQA/SourceTargetDeltas/LeadP/hMixedEventDeltaZvtx", "hMixedEventDeltaZvtx;#Delta Z_{Vtx};Counts", kTH1D, {axisConfigurations.axisMixDeltaZvtx});
      histos.add("EventMixingQA/SourceTargetDeltas/LeadJet/hMixedEventDeltaZvtx", "hMixedEventDeltaZvtx;#Delta Z_{Vtx};Counts", kTH1D, {axisConfigurations.axisMixDeltaZvtx}); // There should be no noticeable change between LeadP and LeadJet's collision-related histograms (if they don't bias event selection)
      histos.add("EventMixingQA/SourceTargetDeltas/SubJet/hMixedEventDeltaZvtx", "hMixedEventDeltaZvtx;#Delta Z_{Vtx};Counts", kTH1D, {axisConfigurations.axisMixDeltaZvtx});

      histos.add("EventMixingQA/SourceTargetDeltas/LeadP/hMixedEventDeltaCent", "hMixedEventDeltaCent;Centrality(%);Counts", kTH1D, {axisConfigurations.axisMixDeltaCentrality});
      histos.add("EventMixingQA/SourceTargetDeltas/LeadJet/hMixedEventDeltaCent", "hMixedEventDeltaCent;Centrality(%);Counts", kTH1D, {axisConfigurations.axisMixDeltaCentrality});
      histos.add("EventMixingQA/SourceTargetDeltas/SubJet/hMixedEventDeltaCent", "hMixedEventDeltaCent;Centrality(%);Counts", kTH1D, {axisConfigurations.axisMixDeltaCentrality});
      
      histos.add("EventMixingQA/SourceTargetDeltas/LeadP/hMixedEventDeltaEta", "hMixedEventDeltaEta;#Delta#eta;Counts", kTH1D, {axisConfigurations.axisMixDeltaEta});
      histos.add("EventMixingQA/SourceTargetDeltas/LeadJet/hMixedEventDeltaEta", "hMixedEventDeltaEta;#Delta#eta;Counts", kTH1D, {axisConfigurations.axisMixDeltaEta});
      histos.add("EventMixingQA/SourceTargetDeltas/SubJet/hMixedEventDeltaEta", "hMixedEventDeltaEta;#Delta#eta;Counts", kTH1D, {axisConfigurations.axisMixDeltaEta});
      
      histos.add("EventMixingQA/SourceTargetDeltas/LeadP/hMixedEventDeltaPhi", "hMixedEventDeltaPhi;#Delta#varphi;Counts", kTH1D, {axisConfigurations.axisMixDeltaPhi});
      histos.add("EventMixingQA/SourceTargetDeltas/LeadJet/hMixedEventDeltaPhi", "hMixedEventDeltaPhi;#Delta#varphi;Counts", kTH1D, {axisConfigurations.axisMixDeltaPhi});
      histos.add("EventMixingQA/SourceTargetDeltas/SubJet/hMixedEventDeltaPhi", "hMixedEventDeltaPhi;#Delta#varphi;Counts", kTH1D, {axisConfigurations.axisMixDeltaPhi});

      // "Closure test" for the binning policy:
      // Every entry's delta must be <= than the width of its own y bin (e.g., a 2-4 GeV/c target cannot borrow from more than 2 GeV/c away).
      // (LeadP only as it would verify the same thing)
      histos.add("EventMixingQA/SourceTargetDeltas/LeadP/h2dMixedEventDeltaPtVsTargetPt", "h2dMixedEventDeltaPtVsTargetPt;#Delta p_{T} (GeV/c);Target proxy p_{T} (GeV/c);Counts", kTH2D, {axisConfigurations.axisMixDeltaPt, axisConfigurations.axisJetPt});
      histos.add("EventMixingQA/SourceTargetDeltas/LeadP/h2dMixedEventDeltaZvtxVsTargetZvtx", "h2dMixedEventDeltaZvtxVsTargetZvtx;#Delta Z_{Vtx} (cm);Target Z_{Vtx} (cm);Counts", kTH2D, {axisConfigurations.axisMixDeltaZvtx, axisConfigurations.axisPVz});
      histos.add("EventMixingQA/SourceTargetDeltas/LeadP/h2dMixedEventDeltaCentVsTargetCent", "h2dMixedEventDeltaCentVsTargetCent;#Delta Centrality (%);Target centrality (%);Counts", kTH2D, {axisConfigurations.axisMixDeltaCentrality, axisConfigurations.axisCentrality});

      // Index Vs "Delta mixed variable" TH2s to check if the width changes with indexing (possible evtSplittings or cannibalizations):
      histos.add("EventMixingQA/IndexQA/LeadP/h2dMixedEventDeltaPtVsDeltaIndex", "h2dMixedEventDeltaPtVsDeltaIndex;#Delta collision index;#Delta p_{T} (GeV/c);Counts", kTH2D, {axisConfigurations.axisMixDeltaIndexCoarse, axisConfigurations.axisMixDeltaPtCoarse});
      histos.add("EventMixingQA/IndexQA/LeadP/h2dMixedEventDeltaZvtxVsDeltaIndex", "h2dMixedEventDeltaZvtxVsDeltaIndex;#Delta collision index;#Delta Z_{Vtx} (cm);Counts", kTH2D, {axisConfigurations.axisMixDeltaIndexCoarse, axisConfigurations.axisMixDeltaZvtxCoarse});
      histos.add("EventMixingQA/IndexQA/LeadP/h2dMixedEventDeltaCentVsDeltaIndex", "h2dMixedEventDeltaCentVsDeltaIndex;#Delta collision index;#Delta Centrality (%);Counts", kTH2D, {axisConfigurations.axisMixDeltaIndexCoarse, axisConfigurations.axisMixDeltaCentCoarse});

      // Counting bit-exact coincidences of values (QA for event mixing to make sure no auto-correlated mixing happens):
      histos.add("EventMixingQA/IdentityChecks/LeadP/hMixedEventIdentityFlags", "hMixedEventIdentityFlags;0: pt, 1: eta, 2: phi, 3: Zvtx;Exact matches", kTH1D, {{4, -0.5f, 3.5f}});
      // Angular coincidence check on a log scale (alternative):
      histos.add("EventMixingQA/IdentityChecks/LeadP/hMixedEventLogAngularSep", "hMixedEventLogAngularSep;log_{10}(#sqrt{#Delta#eta^{2} + #Delta#varphi^{2}});Counts", kTH1D, {{150, -7.f, 0.5f}});
      // Bit-identical Zvtx value check at the LUT building time (characterization of "Birthday paradox" effect):
      histos.add("EventMixingQA/IdentityChecks/hDuplicateZvtxCollisions", "hDuplicateZvtxCollisions;Collisions sharing one Zvtx value;Counts", kTH1D, {{101, 1.5f, 102.5f}});

      // How many mixing candidates a collision actually had:
      histos.add("EventMixingQA/CollLoopOutcome/LeadP/hMixedEventCandidates", "hMixedEventCandidates;Candidates seen by this collision;Counts", kTH1D, {axisConfigurations.axisMixCandidates});
      histos.add("EventMixingQA/CollLoopOutcome/LeadJet/hMixedEventCandidates", "hMixedEventCandidates;Candidates seen by this collision;Counts", kTH1D, {axisConfigurations.axisMixCandidates});
      histos.add("EventMixingQA/CollLoopOutcome/SubJet/hMixedEventCandidates", "hMixedEventCandidates;Candidates seen by this collision;Counts", kTH1D, {axisConfigurations.axisMixCandidates});

      // The same count resolved against the pt bin the collision was mixed in (source and target share it):
      histos.add("EventMixingQA/CollLoopOutcome/LeadP/h2dMixedCandidatesVsPt", "h2dMixedCandidatesVsPt;Candidates;LeadP p_{T} (GeV/c);Counts", kTH2D, {axisConfigurations.axisMixCandidates, axisConfigurations.axisJetPt});
      histos.add("EventMixingQA/CollLoopOutcome/LeadJet/h2dMixedCandidatesVsPt", "h2dMixedCandidatesVsPt;Candidates;LeadJet p_{T} (GeV/c);Counts", kTH2D, {axisConfigurations.axisMixCandidates, axisConfigurations.axisJetPt});
      histos.add("EventMixingQA/CollLoopOutcome/SubJet/h2dMixedCandidatesVsPt", "h2dMixedCandidatesVsPt;Candidates;SubJet p_{T} (GeV/c);Counts", kTH2D, {axisConfigurations.axisMixCandidates, axisConfigurations.axisJetPt});
      // Number of distinct sources stored per target in an event mixing with resampling:
      // (for any value below nProxyResamples, the remaining resampling passes of that collision were empty)
      if (fakePolSwitches.nProxyResamples > 1) {
        const int nPasses = fakePolSwitches.nProxyResamples;
        histos.add("EventMixingQA/CollLoopOutcome/LeadP/hMixedEventDistinctSources", "hMixedEventDistinctSources;Distinct sources stored;Counts", kTH1D, {{nPasses + 1, -0.5, nPasses + 0.5}});
        histos.add("EventMixingQA/CollLoopOutcome/LeadJet/hMixedEventDistinctSources", "hMixedEventDistinctSources;Distinct sources stored;Counts", kTH1D, {{nPasses + 1, -0.5, nPasses + 0.5}});
        histos.add("EventMixingQA/CollLoopOutcome/SubJet/hMixedEventDistinctSources", "hMixedEventDistinctSources;Distinct sources stored;Counts", kTH1D, {{nPasses + 1, -0.5, nPasses + 0.5}});
      }
    }

    // Fetch the X-axes from one of the families (since they all share the same ConfigurableAxis binning)
    mAxisPt = histosRingFamily.get<TH2>(HIST("Ring/DeltaMethod/h2dLambdaPtVsDeltaComp"))->GetXaxis();
    mAxisMass = histosRingFamily.get<TH2>(HIST("Ring/DeltaMethod/h2dMassVsDeltaComp"))->GetXaxis();
    mAxisDTheta = histosRingFamily.get<TH2>(HIST("Ring/DeltaMethod/h2dDeltaThetaVsDeltaComp"))->GetXaxis();
    for (auto const& tracker : {&trackRing, &trackRingKinCuts, &trackJetKinCuts, &trackJetLambdaKinCuts})
      tracker->resize(mAxisPt->GetNbins(), mAxisMass->GetNbins(), mAxisDTheta->GetNbins());
  }

  // Helper to get centrality (same from TableProducer, thanks to templating!):
  template <typename TCollision>
  auto getCentrality(TCollision const& collision)
  {
    if (centralityEstimator == kCentFT0M)
      return collision.centFT0M();
    else if (centralityEstimator == kCentFT0C)
      return collision.centFT0C();
    else if (centralityEstimator == kCentFV0A)
      return collision.centFV0A();
    return -1.f;
  }

  /// \brief Minimal helper to fill the analysis-level cut flow without dealing with bins by hand.
  /// \note CAUTION! If you change the cut order in isV0Accepted, change the label list in init() to match!
  struct AnalysisCutFlowCounter {
    int binValue = -1; // Starts at x=-1: fill() pre-increments, so the first filled bin is always x=0
    HistogramRegistry* histos = nullptr;
    float massV0 = -1.f; // Lambda-like mass of the current V0
    void resetForNewV0(float mass) {
      binValue = -1;
      massV0 = mass;
    }
    void fill() {
      histos->fill(HIST("hAnalysisLevelSelectionV0s"), ++binValue); // Hardcoded names, as they will not change. Increments before filling, by default
      histos->fill(HIST("h2dAnalysisLevelSelectionV0sVsMass"), binValue, massV0);
    }
  };

  AnalysisCutFlowCounter v0AnalysisCutCounter{-1, &histos}; // Any index works here (resetForNewV0 is always called for a new V0 anyways)
 
  /// \brief A function that applies analysisLevelCuts' cuts to select V0s
  template <typename TV0>
  bool isV0Accepted(TV0 const& v0)
  {
    v0AnalysisCutCounter.resetForNewV0(v0.massV0());
    v0AnalysisCutCounter.fill(); // Bin 0: every V0 candidate reaching the analysis level

    const bool isLambda = v0.isLambda();
    histos.fill(HIST("h2dArmenterosInput"), v0.alpha(), v0.qtArm());
 
    // Kinematics:
    if (v0.v0Pt() < analysisLevelCuts.v0MinPt)
      return false;
    v0AnalysisCutCounter.fill();
    if (v0.v0Pt() > analysisLevelCuts.v0MaxPt)
      return false;
    v0AnalysisCutCounter.fill();
    if (std::abs(v0.v0Rapidity()) > analysisLevelCuts.v0MaxRap)
      return false;
    v0AnalysisCutCounter.fill();
    if (std::abs(v0.v0Eta()) > analysisLevelCuts.v0MaxEta)
      return false;
    v0AnalysisCutCounter.fill();
 
    // Armenteros cuts to remove K0s (35% of sample) and possible photons (<1% sample, but cheap to remove):
    if (isLambda) { // Splitting on the Lambda hypothesis in order to properly remove the antiLambda parabola
      if (v0.alpha() < analysisLevelCuts.apAlphaMin)
        return false;
      v0AnalysisCutCounter.fill();

      if (v0.alpha() > analysisLevelCuts.apAlphaMax)
        return false;
      v0AnalysisCutCounter.fill();
    } else {
      if (v0.alpha() > -1*analysisLevelCuts.apAlphaMin)
        return false;
      v0AnalysisCutCounter.fill();

      if (v0.alpha() < -1*analysisLevelCuts.apAlphaMax)
        return false;
      v0AnalysisCutCounter.fill();
    }
    if (v0.qtArm() < analysisLevelCuts.apQtMin)
      return false;
    v0AnalysisCutCounter.fill();
    if (v0.qtArm() > analysisLevelCuts.apQtMax)
      return false;
    v0AnalysisCutCounter.fill();
 
    // TPC-related:
    if (std::abs(v0.prLikeTPCNSigma()) > analysisLevelCuts.nSigmaTPCPrLike)
      return false;
    v0AnalysisCutCounter.fill();
    if (std::abs(v0.piLikeTPCNSigma()) > analysisLevelCuts.nSigmaTPCPiLike)
      return false;
    v0AnalysisCutCounter.fill();
 
    // Topological:
    if (v0.dcaV0Daughters() > analysisLevelCuts.v0MaxDcaDau)
      return false;
    v0AnalysisCutCounter.fill();
    if (v0.v0CosPA() < analysisLevelCuts.v0MinCosPA)
      return false;
    v0AnalysisCutCounter.fill();
    if (v0.v0Radius() < analysisLevelCuts.v0MinRadius)
      return false;
    v0AnalysisCutCounter.fill();
    if (v0.v0Radius() > analysisLevelCuts.v0MaxRadius)
      return false;
    v0AnalysisCutCounter.fill();
 
    // Daughter DCAs to the PV:
    // Datamodel stores these per charge charge, so the proton-like/pion-like mapping needs the hypothesis
    const float dcaPrLikeToPV = std::abs(isLambda ? v0.dcaPosToPV() : v0.dcaNegToPV());
    const float dcaPiLikeToPV = std::abs(isLambda ? v0.dcaNegToPV() : v0.dcaPosToPV());
    if (dcaPrLikeToPV < analysisLevelCuts.v0MinDcaPrLikeToPV)
      return false;
    v0AnalysisCutCounter.fill();
    if (dcaPiLikeToPV < analysisLevelCuts.v0MinDcaPiLikeToPV)
      return false;
    v0AnalysisCutCounter.fill();
 
    // Jets-related (kinematic, quenching, etc.):
    // (TODO)

    if (v0.v0Phi() < v0PhiLimitsVec[0] || v0.v0Phi() > v0PhiLimitsVec[1])
      return false;
    v0AnalysisCutCounter.fill();

    if (v0.massV0() < analysisLevelCuts.v0MassMin)
      return false;
    v0AnalysisCutCounter.fill();

    if (v0.massV0() > analysisLevelCuts.v0MassMax)
      return false;
    v0AnalysisCutCounter.fill();
 
    v0AnalysisCutCounter.fill(); // Final accepted V0s. Redundant with the last cut's bin by construction, kept as a stable reference bin
    histos.fill(HIST("h2dArmenterosAnalysisCuts"), v0.alpha(), v0.qtArm());
    return true;
  }

  // Initializing a random number generator for the worker (for perpendicular-to-jet direction QAs):
  TRandom3 randomGen{0}; // 0 means we auto-seed from machine entropy. This is called once per device in the pipeline, so we should not see repeated seeds across workers
  std::mt19937 rng{std::random_device{}()};

  // Pre-computed values for helper below:
  const double smearSigma = 0.05 * jetR;
  // Normalized eta weights spanning -0.9 to 0.9 (46 bins) and phi weights for the 50 bins spanning [0, 2pi]:
  // (TODO: these are for leading particles, not for jets! Will use them for all proxies for now, as this is mostly QA)
  static constexpr std::array<double, 46> etaLeadPWeights = {{0.01782505198123039, 0.01826119427561306, 0.01890047073124532, 0.01942224199989093, 0.01993380780273602, 0.02047274597515178, 0.02094135547756474, 0.02140259654932778, 0.02178490245182078, 0.02218346916434517, 0.02252861343224298, 0.02278214932340838, 0.02297395476452691, 0.02311861709583109, 0.02322295246943318, 0.02329274166449468, 0.02335344516182264, 0.02335971904087711, 0.02340163522424806, 0.02352868368676468, 0.02345839093195849, 0.02295391718536531, 0.02306543716698383, 0.02293131780181040, 0.02265098126991631, 0.02318893563623931, 0.02322457978177088, 0.02316188601954564, 0.02308812992419636, 0.02305831097751334, 0.02300695504254397, 0.02296398287895598, 0.02286956017362988, 0.02274321888486162, 0.02254049413132752, 0.02233024817234042, 0.02204811997596179, 0.02170252191737713, 0.02130517220864903, 0.02086349641950970, 0.02036590755841183, 0.01987946074337928, 0.01934550418314029, 0.01879487530409137, 0.01812577211265362, 0.01766945805710696}};
  static constexpr std::array<double, 50> phiLeadPWeights = {{0.01907529231698144, 0.02044679008716434, 0.01948618554713157, 0.02046288887443206, 0.02142057576726765, 0.01961361841611185, 0.02174981627752354, 0.02160945937846856, 0.02027231236207667, 0.02153799273983672, 0.02107609996106984, 0.02001899849606885, 0.02196516947939817, 0.02047654705787587, 0.02059561382369167, 0.02148625027035289, 0.02001510925188416, 0.02183059661361331, 0.02111406548114694, 0.01881826666371129, 0.02112797285609031, 0.02034071592473790, 0.01968993216337670, 0.02126946766383166, 0.02025580366897253, 0.02061136834815962, 0.02083881238552183, 0.01994368379135331, 0.02046280212696892, 0.02131631148368759, 0.01967275608960357, 0.02064965975476278, 0.02155758091535052, 0.02012557837329328, 0.02084718400924722, 0.02065094443305587, 0.01969546187428681, 0.02136531489392276, 0.02084491659074202, 0.01970883494847568, 0.02080349992636310, 0.02098440611808174, 0.02159984658204099, 0.02045819959592145, 0.01952755742791563, 0.02166002908586709, 0.02017512590511229, 0.01932658087972190, 0.02108933627170306, 0.02019288778648293}};
  // Build discrete eta distribution for sampling:
  std::discrete_distribution<int> etaLeadPDist{etaLeadPWeights.begin(), etaLeadPWeights.end()}; // Will be passed as the etaDist variable
  std::discrete_distribution<int> phiLeadPDist{phiLeadPWeights.begin(), phiLeadPWeights.end()};

  /// \brief Applies whichever fakePolSwitches distortion is active to a jet-proxy direction, in place. No-op if none are on.
  /// \param proxy input/edited in-place: the proxy's kinematics, overwritten by whichever distortion is active.
  /// \param unitVec input/edited in-place: the proxy direction as a unit vector.
  /// \param minPtThreshold pT threshold for re-evaluating proxy.isValid after distortion.
  /// \param source input only: the previous-jet/mixed-event source proxy. The caller keeps it up to date, not this function.
  ///        (forcePreviousJet fills it from the previous analysed collision's own proxy, doMixedEventProxies from the mixing LUT right before the call)
  /// \param etaDist,phiDist sampling distributions for forceDatalikeJet (drawn from the task's own rng member).
  /// \note  Shared across leadP/leadJet/subJet: the caller resolves which proxy-specific procedure (e.g., LUT for evtMixing) applies.
  /// \note  A missing source proxy invalidates this collision's proxy (the fallback skips the event).
  // Helper to modify the jet direction for QA and for spurious signal baseline removal tests:
  inline void applyProxyDistortion(ProxyKinematics& proxy, XYZVector& unitVec, float minPtThreshold, float maxAbsEta, ProxyKinematics const& source,
                                   std::discrete_distribution<int>& etaDist, std::discrete_distribution<int>& phiDist)
  {
    if (!fakePolSwitches.forcePerpToJet && !fakePolSwitches.forceJetDirectionSmudge && !fakePolSwitches.forceRandJet && !fakePolSwitches.forcePreviousJet && !fakePolSwitches.forceDatalikeJet && !fakePolSwitches.doMixedEventProxies) [[likely]] {
      return; // Skip this function if none of the modifications are actually being executed!
    }

    // "Borrowed" proxies were actually reconstructed in an event.
    // "Artificial" ones create a direction. Only the latter will recalculate pt/eta/phi from the distorted unitVec at the end of this function.
    const bool borrowedPhysicalProxy = fakePolSwitches.forcePreviousJet || fakePolSwitches.doMixedEventProxies;

    // Total momentum of this collision's own proxy, before distortions:
    const double beforeDistTotalMomentum = borrowedPhysicalProxy ? 0. : proxy.pt * std::cosh(proxy.eta); // Only needs to be calculated when borrowedPhysicalProxy is false

    // QA block -- Purposefully changing the jet direction (should kill signal, if any):
    if (fakePolSwitches.forcePerpToJet) {
      // First, we build a vector perpendicular to the jet by picking an arbitrary vector not parallel to the jet
      XYZVector perpVec;
      if (std::abs(unitVec.X()) > 0.99) {
        perpVec = XYZVector(-unitVec.Z(), 0., unitVec.X()).Unit(); // Cross product with Y-axis (0, 1, 0)
      } else {
        perpVec = XYZVector(0., unitVec.Z(), -unitVec.Y()).Unit(); // Cross product with X-axis (1, 0, 0)
      }

      // Now we rotate around the jet axis by a random angle, just to make sure we are not introducing a bias in the QA:
      // We will use Rodrigues' rotation formula (v_rot = v*cos(randomAngle) + (Jet \cross v)*sin(randomAngle))
      const double randomAngle = randomGen.Uniform(0., constants::math::TwoPI);
      unitVec = perpVec * std::cos(randomAngle) + unitVec.Cross(perpVec) * std::sin(randomAngle);
    } else if (fakePolSwitches.forceJetDirectionSmudge) {
      // Smear the jet direction by a small random angle to estimate sensitivity to
      // jet axis uncertainty. We rotate the jet axis by angle theta around a uniformly
      // random perpendicular axis -- this is isotropic and coordinate-independent,
      // unlike smearing eta and phi separately (which would break azimuthal symmetry
      // around the jet axis and depend on where in eta the jet sits).

      // 1) We pick a uniformly random axis perpendicular to the jet.
      // (re-using the same Rodrigues formula as in the forcePerpToJet block above)
      XYZVector perpVec;
      if (std::abs(unitVec.X()) > 0.99) {
        perpVec = XYZVector(-unitVec.Z(), 0., unitVec.X()).Unit(); // Cross product with Y-axis (0, 1, 0)
      } else {
        perpVec = XYZVector(0., unitVec.Z(), -unitVec.Y()).Unit(); // Cross product with X-axis (1, 0, 0)
      }

      // Rotate perpVec around the jet axis by a uniform random azimuth to get
      // a uniformly distributed random perpendicular direction (the smear axis):
      const double smearAzimuth = randomGen.Uniform(0., constants::math::TwoPI);
      XYZVector smearAxis = perpVec * std::cos(smearAzimuth) + unitVec.Cross(perpVec) * std::sin(smearAzimuth);

      // 2) draw the smearing polar angle from a Gaussian:
      // sigma = 0.05 * R --> ~68% of events smeared within 5% of R,
      //                      ~95% of events smeared within 10% of R,
      //                       ~5% see a displacement > 0.1*R (a very "badly determined jet", for our QA purposes)
      // std::abs() folds the symmetric Gaussian onto a half-normal ([0, inf))
      // -- R is not really an angle: just gives me a scale for the angular shift I am performing.
      // -- This may pose problems for forward jets: a small displacement in \theta becomes a large displacement in \eta space
      const double smearAngle = std::abs(randomGen.Gaus(0., smearSigma));

      // 3) rotate the jet axis by smearAngle around smearAxis.
      // Rodrigues is v_rot = v*cos(theta) + (k \cross v)*sin(theta) + k*(k \cdot v)*(1-cos(theta))
      // But the last term vanishes because smearAxis is perpendicular to unitVec:
      unitVec = unitVec * std::cos(smearAngle) + smearAxis.Cross(unitVec) * std::sin(smearAngle);
      // Also, rotation preserves the norm, so no re-normalisation is needed for this to be a unit vector.
    } else if (fakePolSwitches.forceRandJet) {
      // This randomization was made different for each proxy (LeadP, LeadJet, SubLeadJet): bear that in mind!
      // 1) Uniformly sample cos(theta) and phi to ensure an isotropic distribution (could also use TRandom::Sphere as well, but may be slower)
      // Notice that uniformly sampling theta would make the distribution non-isotropic, thus we use cos(theta)!
      const double cosTheta = randomGen.Uniform(-1., 1.);
      const double sinTheta = std::sqrt(1. - cosTheta * cosTheta);
      const double randPhi = randomGen.Uniform(0., constants::math::TwoPI);

      // 2) Construct the new random unit vector (there is no need to use the magnitude at all! We only need direction here):
      unitVec = XYZVector(sinTheta * std::cos(randPhi), sinTheta * std::sin(randPhi), cosTheta);
    } else if (borrowedPhysicalProxy) { // forcePreviousJet or doMixedEventProxies: a real proxy reconstructed in another collision
      if (source.isValid) {
        // Adopt the source proxy wholesale:
        // From here on this collision is treated as if it had had that leading particle/jet all along.
        proxy.pt = source.pt;
        proxy.eta = source.eta;
        proxy.phi = source.phi;
        const double inverseCoshEta = 1.0 / std::cosh(source.eta);
        const double sinPhi = std::sin(source.phi);
        const double cosPhi = std::cos(source.phi);
        unitVec = XYZVector(cosPhi * inverseCoshEta, sinPhi * inverseCoshEta, std::tanh(source.eta));
      } else {
        // No source proxy for this collision: the previous collision had none, or the mixing bins were too sparse.
        // Either way this collision cannot be used.
        proxy.isValid = false;
      }
    } else if (fakePolSwitches.forceDatalikeJet) { // A compromise between forceRandJet and forcePreviousJet, using data-like weights for sampling jets
      const float etaMin = -0.92f;
      const float etaBinWidth = 0.04f;
      constexpr float phiBinWidth = constants::math::TwoPI / 50.f;

      // Pick one of the 46 bins according to etaWeights:
      const int binEtaIdx = etaDist(rng);
      const int binPhiIdx = phiDist(rng);

      // Uniformly smear inside the chosen bin:
      proxy.eta = etaMin + etaBinWidth * (binEtaIdx + std::generate_canonical<float, 24>(rng));
      proxy.phi = phiBinWidth * (binPhiIdx + std::generate_canonical<float, 24>(rng));

      const double inverseCoshEta = 1.0 / std::cosh(proxy.eta);
      const double sinPhi = std::sin(proxy.phi);
      const double cosPhi = std::cos(proxy.phi);
      unitVec = XYZVector(cosPhi * inverseCoshEta, sinPhi * inverseCoshEta, std::tanh(proxy.eta));
    }

    if (proxy.isValid) { // If you don't check this flag here, the borrowed-proxy miss above would be silently overwritten
      // if (borrowedPhysicalProxy) {
      //   // The adopted proxy is a real one that already passed this same cut when the pool was built
      //   // (minimum-pT gates on the mixing LUTs, isValid on the previous collision).
      //   // Re-checked only as a guard: it should never fire.
      //   proxy.isValid = proxy.pt > minPtThreshold;
      // }
      // Artificial proxy handling -- only the direction was "invented", so pT, phi and eta are rebuilt from unitVec.
      // (Without this, later kinematic selections and QAs are inconsistent with the calculated ring observable)
      if (!borrowedPhysicalProxy) {
        // For stability (Rho is the projection on the transverse plane, badly behaved for high |eta|):
        const double transverseNorm = std::max(unitVec.Rho(), 1e-12); // Stability guard

        // Our choice is preserving |p| across the change of direction, so pT follows the new polar angle:
        proxy.pt = beforeDistTotalMomentum * transverseNorm;

        if (!fakePolSwitches.forceDatalikeJet) { // DatalikeJet doesn't require recalculating these directions
          // Recalculate phi:
          proxy.phi = RecoDecay::constrainAngle(std::atan2(unitVec.Y(), unitVec.X()), 0.0f); // atan2 outputs [-PI, PI), and DataModel convention was [0,2PI) as per FastJet's phi() getter

          // Stable eta computation:
          // Stabler than 0.5 * std::log((1. + cosTheta) / (1. - cosTheta))
          // (for forceDatalikeJet this reproduces the eta/phi its own branch sampled, so one path covers all four modes)
          proxy.eta = std::asinh(unitVec.Z() / transverseNorm);
        }

        // These are not jets we measured, so the pT selection is optional here: it exists to let the bias
        // it introduces be measured.
        if (fakePolSwitches.gatePtOnArtificialProxies)
          proxy.isValid = proxy.pt > minPtThreshold;
        // forceRandJet draws isotropic jets in solid angle, so this puts all "artificial" modes under data-like acceptance:
        if (fakePolSwitches.gateEtaOnArtificialProxies)
          proxy.isValid = std::abs(proxy.eta) < maxAbsEta;
      }
    }
  }

  /// \brief The ring observable of one V0 with respect to one proxy, plus everything derived from the same cross product.
  /// \param proxyUnitVec the proxy direction as a unit vector (after applyProxyDistortion).
  /// \param proxyPhi the proxy azimuth, for deltaPhi.
  /// \param v0Inputs the per-V0 quantities, computed once and shared by every proxy.
  /// \note  R = (3/alpha) p_D_hat^* \cdot n, with n = (t_hat x p_Lambda)/|t_hat x p_Lambda|. The longitudinal modes keep only
  ///        the beam-axis part of p_D^* (R_z) or that part made transverse to the Lambda (R_z^T).
  /// \note  Call it only for a valid proxy: a missing proxy keeps the ProxyRingCache defaults.
  ProxyRingCache computeProxyRing(const XYZVector& proxyUnitVec, float proxyPhi, const V0RingInputs& v0Inputs)
  {
    ProxyRingCache ring;
    XYZVector cross = proxyUnitVec.Cross(v0Inputs.lambdaLike3Vec);
    const float invCrossNorm = 1.f / cross.R(); // Caching the .R() result
    const float ringFullUnit = v0Inputs.protonLikeStarUnit3Vec.Dot(cross) * invCrossNorm; // v: the unitless full ring
    if (!longitudinalRingMode) {
      ring.ringObservable = ringFullUnit;
    } else {
      ring.nz = cross.Z() * invCrossNorm;
      ring.ringObservable = v0Inputs.protonStarRingBeamComponent * ring.nz;
      ring.kappaDen = ring.nz * ring.nz;
      // Diagnostics only: the complement R - R_long, the helicity term R_z - R_z^T, and the bearing chi:
      ring.ringPerp = v0Inputs.ringPrefactor * (ringFullUnit - ring.ringObservable);
      ring.helicityTerm = v0Inputs.ringPrefactor * v0Inputs.protonStarHelicityBeamComponent * ring.nz;
      ring.chi = std::atan2(-cross.Z() * v0Inputs.v0p, cross.Y() * v0Inputs.v0px - cross.X() * v0Inputs.v0py);
    }
    ring.kappaNum = ring.ringObservable * ringFullUnit; // u v (u^2 for the full ring)
    // Adding the prefactor related to the CP-violating decay (decay constants have different signs). Prefactor goes last!
    ring.ringObservable *= v0Inputs.ringPrefactor;
    // Angular variables
    ring.deltaPhi = wrapToPiFast(v0Inputs.v0phi - proxyPhi); // Wrapped to [-PI, PI), for convenience
    ring.cosDeltaTheta = proxyUnitVec.Dot(v0Inputs.lambdaLikeUnit3Vec); // Uses the pre-calculated unit vectors to avoid recomputation
    ring.deltaTheta = std::acos(ring.cosDeltaTheta);                     // 3D angular separation. Same as ROOT::Math::VectorUtil::Angle(proxyUnitVec, lambdaLike3Vec);
    return ring;
  }

  /// \brief Uniform random pick of nProxyResamples DISTINCT sources among each target's candidate window, via reservoir sampling
  ///        (Uses Algorithm R with nProxyResamples slots, without replacement, as a repeated source adds no new jet direction and would only shrink the error bars artificially)
  ///        (cannibalization of proxies by neighbouring collisions in Continuous Readout is not a worry as ITS hits are being demanded)
  /// \note  Reuses the task's own rng member rather than a separate generator per proxy.
  /// \note  Draws even while the reservoir is still filling with candidates for mixing, so nProxyResamples = 1 is the single-slot rule, draw for draw.
  void reservoirInsert(ProxyByCollision& proxyByCollision, size_t targetSlot, int64_t targetId, size_t sourceSlot, int64_t sourceId)
  {
    // A collision must never borrow from itself:
    // (StrictlyUpperSameIndexPolicy should make this impossible, but we check nonetheless)
    if (targetId == sourceId)
      LOG(fatal) << "EventMixing: collision " << targetId << " offered itself as a mixing source.";
    const size_t nResamples = static_cast<size_t>(fakePolSwitches.nProxyResamples);
    const int32_t nSeen = ++proxyByCollision.candidateCountVec[targetSlot];
    std::uniform_int_distribution<int> pick(0, nSeen - 1);
    const size_t drawn = static_cast<size_t>(pick(rng));
    if (static_cast<size_t>(nSeen) <= nResamples) // Reservoir not full yet: fill the next slot directly.
      proxyByCollision.mixedSourcesVec[targetSlot * nResamples + (nSeen - 1)] = static_cast<int32_t>(sourceSlot);
    else if (drawn < nResamples)
      // Reservoir full: Algorithm R.
      proxyByCollision.mixedSourcesVec[targetSlot * nResamples + drawn] = static_cast<int32_t>(sourceSlot);
  }

  /// \brief How many different target collisions each source collision ended up supplying, from a finished LUT.
  /// \note  A pair is visited once per pool, so a source appears at most once among a target's entries: entries counted = targets.
  /// \note  Keyed by source slot: sources are sparse, so a map is the right shape here even though the LUT is optimized as a vector.
  static std::unordered_map<int32_t, int> tallySourceUsage(std::vector<int32_t> const& lut)
  {
    std::unordered_map<int32_t, int> usage;
    for (const int32_t sourceSlot : lut) {
      if (sourceSlot >= 0)
        usage[sourceSlot]++;
    }
    return usage;
  }

  /// \brief The proxy a target collision borrows in one resampling pass, rebuilt from the source collision's own caches.
  /// \return An empty MixedProxyInfo (proxy.isValid = false, sourceCollisionId = -1) on an empty LUT entry.
  /// \note  Only valid proxies are pooled, so a stored source always carries proxy.isValid = true.
  MixedProxyInfo rebuildBorrowedProxyFromCache(ProxyByCollision const& proxyByCollision, std::vector<MixingAxes> const& mixingAxesByCollision,
                               int64_t collisionIndexBase, size_t targetSlot, size_t pass)
  {
    const size_t nResamples = static_cast<size_t>(fakePolSwitches.nProxyResamples);
    const int32_t sourceSlot = proxyByCollision.mixedSourcesVec[targetSlot * nResamples + pass];
    if (sourceSlot < 0)
      return MixedProxyInfo{};
    auto const& axes = mixingAxesByCollision[sourceSlot];
    return MixedProxyInfo{proxyByCollision.proxyKinematicsVec[sourceSlot], axes.zvtx, axes.centrality, collisionIndexBase + sourceSlot};
  }

  /// \brief Distinct sources a target stored, i.e. how many of its resampling passes borrow (entries fill contiguously from the first).
  size_t distinctSourcesOf(ProxyByCollision const& proxyByCollision, size_t targetSlot)
  {
    return std::min(static_cast<size_t>(proxyByCollision.candidateCountVec[targetSlot]), static_cast<size_t>(fakePolSwitches.nProxyResamples));
  }

  /// \brief Aggregates every collision with a valid proxy by its (Zvtx, proxy pT, centrality) mixing bin.
  /// \note  Collisions are visited in ascending index order, so each pool/window already comes out sorted by collision index.
  std::vector<std::vector<int32_t>> poolByBin(BinningPolicyBase<3>& mixBinning, std::vector<ProxyKinematics> const& proxyKinematicsVec,
                                              std::vector<MixingAxes> const& mixingAxesByCollision)
  {
    std::vector<std::vector<int32_t>> pools(mixBinning.getAllBinsCount()); // Method from BinningPolicy.h
    for (size_t slot = 0; slot < proxyKinematicsVec.size(); ++slot) {
      if (!proxyKinematicsVec[slot].isValid)
        continue;
      auto const& axes = mixingAxesByCollision[slot];
      const int bin = mixBinning.getBin(std::make_tuple(axes.zvtx, proxyKinematicsVec[slot].pt, axes.centrality));
      if (bin >= 0)
        pools[bin].push_back(static_cast<int32_t>(slot));
    }
    return pools;
  }

  // Defining filters for events:
  Filter zvtxFilter = (nabs(o2::aod::lambdajetpol::zvtx) < maxZVtxPosition);

  // Preslices for correct collisions association:
  // (tested custom grouping and performs worse here)
  Preslice<aod::RingJets> perColJets = o2::aod::lambdajetpol::ringCollisionId;
  Preslice<aod::RingLaV0s> perColV0s = o2::aod::lambdajetpol::ringCollisionId;
  Preslice<aod::RingLeadPs> perColLeadPs = o2::aod::lambdajetpol::ringCollisionId;
  /// \brief Main analysis loop: for each collision, rebuilds the leading jet/particle proxies (with optional fakePolSwitches distortions), then loops over V0s computing the ring observable and polarization-vector profiles for every enabled kinematic-cut family.
  void processPolarizationData(soa::Filtered<o2::aod::RingCollisions> const& collisions, o2::aod::RingJets const& jets, o2::aod::RingLaV0s const& v0s,
                               o2::aod::RingLeadPs const& leadPs)
  {
    // Caches for borrowed physical jets (previous jet event mixing, or full window event mixing):
    SetOfProxies prevJetCache{};    // Carried between collisions. Default-initialized, so every isValid starts as false
    SetOfProxies mixedProxyCache{}; // Refilled per collision from the mixing LUTs
    // As forcePreviousJet and doMixedEventProxies are mutually exclusive we can bind them to a single variable:
    SetOfProxies& proxyCache = fakePolSwitches.doMixedEventProxies ? mixedProxyCache : prevJetCache;
    // forcePreviousJet cannot be resampled (fatal in init), while doMixedEventProxies reads one LUT entry per resampling pass.

    // Building vectors for event mixing and leading/subleading jet finding:
    int64_t collisionIndexBase = 0;
    size_t collisionIndexExtent = 0;
    {
      bool isFirst = true;
      for (auto const& collision : collisions) {
        const int64_t idx = collision.globalIndex();
        if (isFirst) {
          collisionIndexBase = idx;
          isFirst = false;
        }
        collisionIndexExtent = static_cast<size_t>(idx - collisionIndexBase) + 1;
      }
    }
    // globalIndex() on a Filtered iterator is the row in the *unfiltered* table, so it can exceed collisions.size():
    // (Take the real extent and offset by the first index, so the vectors stay as small as the data allows!)
    auto slotOf = [collisionIndexBase, collisionIndexExtent](int64_t globalIdx) {
      const int64_t slot = globalIdx - collisionIndexBase;
      if (slot < 0 || static_cast<size_t>(slot) >= collisionIndexExtent)
        LOG(fatal) << "EventMixing: collision index " << globalIdx << " outside the dataframe extent [" << collisionIndexBase << ", " << collisionIndexBase + static_cast<int64_t>(collisionIndexExtent) << ").";
      return static_cast<size_t>(slot);
    };

    // Direct-indexed instead of hashed (vectors instead of unordered_maps):
    // (at 1E6 collisions per dataframe, unordered_map did many scattered allocations and lookup ended up with a costly DRAM access)
    std::vector<MixingAxes> mixingAxesByCollision(collisionIndexExtent);
    // doMixedEventProxies caches: three independent mixings, one per proxy (a collision may have a valid LeadP, but no SubLeadJet).
    // Mixing is binned on (Zvtx, proxy pt, centrality), varying only eta/phi.
    // Each target owns nProxyResamples consecutive entries, so we only write the indices instead of copying MixedProxyInfo as in previous implementations.
    const size_t nResamples = static_cast<size_t>(fakePolSwitches.nProxyResamples);
    const size_t mixingLutSize = fakePolSwitches.doMixedEventProxies ? collisionIndexExtent * nResamples : 0;
    ProxyByCollision leadPByCollision, leadJetByCollision, subJetByCollision;
    for (auto const& proxyByCollision : {&leadPByCollision, &leadJetByCollision, &subJetByCollision}) {
      proxyByCollision->proxyKinematicsVec.assign(collisionIndexExtent, ProxyKinematics{});
      proxyByCollision->mixedSourcesVec.assign(mixingLutSize, -1);
      proxyByCollision->candidateCountVec.assign(collisionIndexExtent, 0); // Candidate counters for the mixing: main loop below reads them back for QA
    }

    // Leading/subleading jet per collision, resolved once here instead of inside the (possibly nProxyResamples times resampled) main loop.
    // Both the main loop and the leadJet/subJet mixing functions read from this, so RingJets is scanned exactly once
    for (auto const& collision : collisions) {
      const auto collId = collision.globalIndex();
      const size_t slot = slotOf(collId);
      ProxyKinematics leadJet;
      ProxyKinematics subJet;

      // Cached here so the pooling below never needs a collision iterator:
      mixingAxesByCollision[slot] = {collision.zvtx(), getCentrality(collision)};

      // std::optional avoids undefined behaviour from a default-constructed iterator:
      std::optional<o2::aod::RingJets::iterator> leadJetRow;
      std::optional<o2::aod::RingJets::iterator> subJetRow;
      for (auto const& jet : jets.sliceBy(perColJets, collId)) {
        const auto jetpt = jet.jetPt();
        if (jetpt > leadJet.pt) {
          // Current leading becomes subleading:
          subJet.pt = leadJet.pt;
          subJetRow = leadJetRow; // may still be std::nullopt on first pass -- that is fine!
          // Now update the leading jet:
          leadJet.pt = jetpt;
          leadJetRow = jet;
        } else if (jetpt > subJet.pt) { // Update subleading only:
          subJet.pt = jetpt;
          subJetRow = jet;
        }
      }
      // Finer control on jet momentum, further than TableProducer pre-selection:
      leadJet.isValid = leadJet.pt > minLeadJetPt;
      subJet.isValid = subJet.pt > minSubLeadJetPt;
      if (leadJet.isValid) {
        leadJet.eta = leadJetRow->jetEta();
        leadJet.phi = leadJetRow->jetPhi();
      }
      if (subJet.isValid) {
        subJet.eta = subJetRow->jetEta();
        subJet.phi = subJetRow->jetPhi();
      }
      leadJetByCollision.proxyKinematicsVec[slot] = leadJet;
      subJetByCollision.proxyKinematicsVec[slot] = subJet;

      // The leading particle is resolved here too, so the pooling reads a vector instead of slicing RingLeadPs.
      // A collision with no leadP simply keeps isValid = false, and its mixing axes are stored regardless
      for (auto const& lp : leadPs.sliceBy(perColLeadPs, collId)) {
        leadPByCollision.proxyKinematicsVec[slot] = {lp.leadParticlePt() > minLeadParticlePt, lp.leadParticlePt(), lp.leadParticleEta(), lp.leadParticlePhi()};
        break; // Table has at most one LeadP per collision, but we break nonetheless
      }
    }

    // A small guard:
    const bool doMixingQA = fakePolSwitches.doMixedEventProxies && qaSwitches.doEventMixingQA;
    const bool doResamplingPassQA = fakePolSwitches.nProxyResamples > 1; // Its histograms are only booked when this is true

    // Bit-identical check on Zvtx values (QA for event mixing):
    if (doMixingQA) {
      std::unordered_map<float, int> zvtxOccupancy;
      for (auto const& collision : collisions)
        ++zvtxOccupancy[collision.zvtx()];
      for (auto const& kv : zvtxOccupancy) {
        if (kv.second > 1)
          histos.fill(HIST("EventMixingQA/IdentityChecks/hDuplicateZvtxCollisions"), kv.second);
      }
    }
    // First we build lookup tables based on current dataframe's collisions (connects pairs of jet proxies from similar collisions):
    // (these proxies may come from collisions with no valid Lambdas, by construction, enabling more mixes)
    // (Built once, out of the resampling loop: each target keeps up to nProxyResamples distinct sources, one per resampling pass)
    if (fakePolSwitches.doMixedEventProxies) {
      const auto tLutStart = std::chrono::steady_clock::now();

      // All three proxies bin on the same (Zvtx, proxy pT, centrality) axes, so a single policy object serves them, even though proxy pT varies:
      // We implement using methods straight from BinningPolicy.h, such as getBin() and getAllBinsCount()
      BinningPolicyBase<3> mixBinning{{axisConfigurations.axisPVz, axisConfigurations.axisJetPt, axisConfigurations.axisCentrality}, true}; // Ignore overflows (pT guards overflow to -999.f by construction)

      // Counting what actually enters the mixing (not what enters the dataframe), before any pooling starts:
      int64_t nValidLeadP = 0, nValidLeadJet = 0, nValidSubJet = 0;
      for (size_t slot = 0; slot < collisionIndexExtent; ++slot) {
        nValidLeadP += leadPByCollision.proxyKinematicsVec[slot].isValid ? 1 : 0;
        nValidLeadJet += leadJetByCollision.proxyKinematicsVec[slot].isValid ? 1 : 0;
        nValidSubJet += subJetByCollision.proxyKinematicsVec[slot].isValid ? 1 : 0;
      }
      LOG(info) << "LUT input: " << collisions.size() << " collisions, extent " << collisionIndexExtent << ", validLeadP " << nValidLeadP << ", validLeadJet " << nValidLeadJet << ", validSubJet " << nValidSubJet;

      /// \brief Pairs each collision with the next mixedEventWindowSize partners inside its own bin, never itself.
      /// \note  Reproduces CombinationsBlockStrictlyUpperSameIndexPolicy exactly, in the same bins, same forward window,
      ///        strictly upper. The advantage is that it runs in O(M*W) rather than the O(M^2) though, M being the collisions carrying a valid proxy.
      /// \note  onPair owns the reservoir inserts and the QA fills, because HIST() needs literal names and so the fills cannot be parameterized here.
      auto pairWithinWindows = [&](std::vector<std::vector<int32_t>> const& pools, auto&& onPair) {
        int64_t nPairs = 0;
        for (auto const& pool : pools) {
          for (size_t a = 0; a < pool.size(); ++a) {
            const size_t slot1 = pool[a];
            const int64_t id1 = collisionIndexBase + static_cast<int64_t>(slot1);
            const size_t lastPartner = std::min(a + static_cast<size_t>(fakePolSwitches.mixedEventWindowSize) + 1, pool.size());
            for (size_t b = a + 1; b < lastPartner; ++b) {
              const size_t slot2 = pool[b];
              const int64_t id2 = collisionIndexBase + static_cast<int64_t>(slot2);
              // Continuous-readout cannibalisation test:
              // (should not be impactful with gap ~ 10, as close-index collisions are most likely not close in time on the derived data, but kept as QA)
              if (fakePolSwitches.mixingIdxGapSize > 0 && std::abs(id2 - id1) <= fakePolSwitches.mixingIdxGapSize)
                continue;
              onPair(slot1, id1, slot2, id2);
              ++nPairs;
            }
          }
        }
        return nPairs;
      };

      /// \brief Forward window occupancy for one collision, i.e. how many partners it is about to be paired with.
      /// \note  Should reproduce currentWindowNeighbours(), saturating at mixedEventWindowSize.
      auto windowNeighboursOf = [&](std::vector<std::vector<int32_t>> const& pools, auto&& onCollision) {
        for (auto const& pool : pools) {
          for (size_t a = 0; a < pool.size(); ++a)
            onCollision(std::min(static_cast<size_t>(fakePolSwitches.mixedEventWindowSize), pool.size() - a - 1));
        }
      };

      // Starting the window loops for each jet proxy:
      // Leading particle:
      const auto leadPPools = poolByBin(mixBinning, leadPByCollision.proxyKinematicsVec, mixingAxesByCollision);
      if (doMixingQA) {
        windowNeighboursOf(leadPPools, [&](size_t nNeighbours) { histos.fill(HIST("EventMixingQA/LeadP/hMixedEventWindowNeighbours"), nNeighbours); });
      }
      const int64_t nPairsLeadP = pairWithinWindows(leadPPools, [&](size_t slot1, int64_t id1, size_t slot2, int64_t id2) {
        // Both sides of the pair feed each other's reservoir, properly using what is already in memory. This also lets a collision borrow from
        // earlier collisions as well, eliminating the forward-only bias of the sliding window in the event mixing procedure.
        reservoirInsert(leadPByCollision, slot1, id1, slot2, id2);
        reservoirInsert(leadPByCollision, slot2, id2, slot1, id1);
        if (doMixingQA)
          histos.fill(HIST("EventMixingQA/IndexQA/LeadP/hMixedEventDeltaIndexEligible"), id2 - id1);
      });
      const auto tLeadPDone = std::chrono::steady_clock::now();
      LOG(info) << "  leadP loop: " << std::chrono::duration<double>(tLeadPDone - tLutStart).count() << " s, " << nPairsLeadP << " pairs";

      // Leading jet:
      const auto leadJetPools = poolByBin(mixBinning, leadJetByCollision.proxyKinematicsVec, mixingAxesByCollision);
      if (doMixingQA) {
        windowNeighboursOf(leadJetPools, [&](size_t nNeighbours) { histos.fill(HIST("EventMixingQA/LeadJet/hMixedEventWindowNeighbours"), nNeighbours); });
      }
      const int64_t nPairsLeadJet = pairWithinWindows(leadJetPools, [&](size_t slot1, int64_t id1, size_t slot2, int64_t id2) {
        reservoirInsert(leadJetByCollision, slot1, id1, slot2, id2);
        reservoirInsert(leadJetByCollision, slot2, id2, slot1, id1);
        if (doMixingQA)
          histos.fill(HIST("EventMixingQA/IndexQA/LeadJet/hMixedEventDeltaIndexEligible"), id2 - id1);
      });
      const auto tLeadJetDone = std::chrono::steady_clock::now();
      LOG(info) << "  leadJet loop: " << std::chrono::duration<double>(tLeadJetDone - tLeadPDone).count() << " s, " << nPairsLeadJet << " pairs";

      // Subleading jet:
      const auto subJetPools = poolByBin(mixBinning, subJetByCollision.proxyKinematicsVec, mixingAxesByCollision);
      if (doMixingQA) {
        windowNeighboursOf(subJetPools, [&](size_t nNeighbours) { histos.fill(HIST("EventMixingQA/SubJet/hMixedEventWindowNeighbours"), nNeighbours); });
      }
      const int64_t nPairsSubJet = pairWithinWindows(subJetPools, [&](size_t slot1, int64_t id1, size_t slot2, int64_t id2) {
        reservoirInsert(subJetByCollision, slot1, id1, slot2, id2);
        reservoirInsert(subJetByCollision, slot2, id2, slot1, id1);
        if (doMixingQA)
          histos.fill(HIST("EventMixingQA/IndexQA/SubJet/hMixedEventDeltaIndexEligible"), id2 - id1);
      });
      const auto tSubJetDone = std::chrono::steady_clock::now();
      LOG(info) << "  subJet loop: " << std::chrono::duration<double>(tSubJetDone - tLeadJetDone).count() << " s, " << nPairsSubJet << " pairs";


      // Source-usage QA, over every stored entry (one per borrowing/resampling pass. An entry's target is entry / nResamples):
      if (doMixingQA) {
        for (size_t entry = 0; entry < leadPByCollision.mixedSourcesVec.size(); ++entry) {
          const int32_t sourceSlot = leadPByCollision.mixedSourcesVec[entry];
          if (sourceSlot < 0) continue;
          histos.fill(HIST("EventMixingQA/IndexQA/LeadP/hMixedEventDeltaIndexSelected"), static_cast<int64_t>(entry / nResamples) - sourceSlot);
        }
        for (size_t entry = 0; entry < leadJetByCollision.mixedSourcesVec.size(); ++entry) {
          const int32_t sourceSlot = leadJetByCollision.mixedSourcesVec[entry];
          if (sourceSlot < 0) continue;
          histos.fill(HIST("EventMixingQA/IndexQA/LeadJet/hMixedEventDeltaIndexSelected"), static_cast<int64_t>(entry / nResamples) - sourceSlot);
        }
        for (size_t entry = 0; entry < subJetByCollision.mixedSourcesVec.size(); ++entry) {
          const int32_t sourceSlot = subJetByCollision.mixedSourcesVec[entry];
          if (sourceSlot < 0) continue;
          histos.fill(HIST("EventMixingQA/IndexQA/SubJet/hMixedEventDeltaIndexSelected"), static_cast<int64_t>(entry / nResamples) - sourceSlot);
        }

        // One fill per source, at its own eta/phi (the slot reaches its cache directly, so no LUT re-scan is needed):
        for (auto const& kv : tallySourceUsage(leadPByCollision.mixedSourcesVec)) {
          auto const& source = leadPByCollision.proxyKinematicsVec[kv.first];
          histos.fill(HIST("EventMixingQA/SourceUsage/LeadP/hMixedEventSourceUsageCount"), kv.second);
          histos.fill(HIST("EventMixingQA/SourceUsage/LeadP/pMixedEventSourceUsageVsEta"), source.eta, kv.second);
          histos.fill(HIST("EventMixingQA/SourceUsage/LeadP/pMixedEventSourceUsageVsPhi"), source.phi, kv.second);
        }
        for (auto const& kv : tallySourceUsage(leadJetByCollision.mixedSourcesVec)) {
          auto const& source = leadJetByCollision.proxyKinematicsVec[kv.first];
          histos.fill(HIST("EventMixingQA/SourceUsage/LeadJet/hMixedEventSourceUsageCount"), kv.second);
          histos.fill(HIST("EventMixingQA/SourceUsage/LeadJet/pMixedEventSourceUsageVsEta"), source.eta, kv.second);
          histos.fill(HIST("EventMixingQA/SourceUsage/LeadJet/pMixedEventSourceUsageVsPhi"), source.phi, kv.second);
        }
        for (auto const& kv : tallySourceUsage(subJetByCollision.mixedSourcesVec)) {
          auto const& source = subJetByCollision.proxyKinematicsVec[kv.first];
          histos.fill(HIST("EventMixingQA/SourceUsage/SubJet/hMixedEventSourceUsageCount"), kv.second);
          histos.fill(HIST("EventMixingQA/SourceUsage/SubJet/pMixedEventSourceUsageVsEta"), source.eta, kv.second);
          histos.fill(HIST("EventMixingQA/SourceUsage/SubJet/pMixedEventSourceUsageVsPhi"), source.phi, kv.second);
        }
      }
      LOG(info) << "LUT build: " << std::chrono::duration<double>(std::chrono::steady_clock::now() - tLutStart).count() << " s for " << collisions.size() << " collisions";
    }

    for (int idxResampling = 0; idxResampling < fakePolSwitches.nProxyResamples; idxResampling++) { // resampling loop for forceRandJet, forceDatalikeJet and doMixedEventProxies
      for (auto const& collision : collisions) {
        if (fakePolSwitches.doMixedEventProxies) { // Making sure cache is properly reset for each new collision
          proxyCache = {};
        }
        const float collisionPVz = collision.zvtx();

        const auto collId = collision.globalIndex(); // The self-index accessor
        const size_t collSlot = slotOf(collId);       // This collision's slot in the per-collision caches
        const float centrality = getCentrality(collision);

        // Used this dummy for backwards compatibility, under the reasonable assumption that the field points always in the same *direction* in the used runs
        // (it is not worth it to fetch and store the magnetic field in the datamodel)
        const float magField = 1.f; // Purely geometric.

        // Slice jets, V0s and leading particle belonging to this collision:
        // (global collision indices repeat a lot, but they are unique to a same TimeFrame (TF) subfolder in the derived data)
        auto v0sInColl = v0s.sliceBy(perColV0s, collId);
        auto leadPsInColl = leadPs.sliceBy(perColLeadPs, collId);

        // Check if there is at least one V0 and one jet in the collision:
        // (in the way I fill the table, there is always at least one V0 in
        //  the stored collision, but the jets table can not be filled for
        //  that collision, and a collision may not be filled when the jets
        //  table is. Be mindful of that!)
        // 1) Require at least one V0:
        const int nLambdaLikeV0s = v0sInColl.size(); // Caching this variable, as it will be reused in the loop
                                                     // In the latest datamodel format, only unambiguous V0s (Lambda XOR antiLambda) are saved,
                                                     // so the number of V0s in the collision table is the number of Lambdas/antiLambda identified
                                                     // by the table producer.
        if (!nLambdaLikeV0s)
          continue;

        // 2) We require at least one leading particle:
        // (The goal is to see how diluted the signal gets with events which don't even have a loose FastJet jet)
        // (The leading particle is built from all tracks that passed the pseudojet
        // selection, so it exists whenever FastJet was run on this collision.
        // Events that have a leading jet always have a leading particle too, but
        // the converse is not true: events can have a leading particle with no jet
        // if no jet survives the pT threshold/the background subtraction)
        // (At least that is the case when minLeadParticlePt = 0)
        ProxyKinematics leadP; // Initialized with the default guard values (see ProxyKinematics)
        float leadPPx = 0., leadPPy = 0., leadPPz = 0.;
        for (auto const& lp : leadPsInColl) {
          // Table should contain exactly one entry per collision, but we break immediately to be safe:
          leadP.pt = lp.leadParticlePt();
          leadP.eta = lp.leadParticleEta();
          leadP.phi = lp.leadParticlePhi();
          // Using dynamic columns to make code cleaner:
          leadPPx = lp.leadParticlePx();
          leadPPy = lp.leadParticlePy();
          leadPPz = lp.leadParticlePz();
        }
        // Apply minimum pT selection for the leading particle (not necessarily the same as in derived data builder. Can be a stricter cut!):
        leadP.isValid = leadP.pt > minLeadParticlePt;

        // Snapshot this collision's own leading particle before any distortion can overwrite it,
        // so forcePreviousJet can hand it to the next collision:
        const ProxyKinematics beforeDistLeadP = leadP;

        // Build leading particle unit vector, outside the V0 loop for performance.
        XYZVector leadPUnitVec(1., 0., 0.); // dummy (overwritten below when leadP.isValid)
        float leadPZ = 0.;                  // \hat t_z of the leading-particle proxy (see axisLeadPZ)
        if (leadP.isValid) {
          leadPUnitVec = XYZVector(leadPPx, leadPPy, leadPPz).Unit();
          leadPZ = leadPUnitVec.Z();

          // doMixedEventProxies: get this collision's mixing partner (if any) from the LUT built above:
          // (this check is performed only if leadP.isValid is true for performance, as the LeadPPt matching for the event mixing would also demand a minimum jet pT)
          // (it is also a physics selection: we want to mix the correlations in events that could actually have a ring formed)
          if (fakePolSwitches.doMixedEventProxies) {
            const MixedProxyInfo mixLeadP = rebuildBorrowedProxyFromCache(leadPByCollision, mixingAxesByCollision, collisionIndexBase, collSlot, idxResampling); // Fetching information for this collId's event mixing source
            proxyCache.leadP = mixLeadP.proxy; // Empty entry (isValid = false): no valid mixing target, or fewer distinct sources than resampling passes

            // Declaring bools before the proxyCache.leadP.isValid checks as they will be used inside the doMixingQA block too:
            bool samePtProxyLeadP = false;
            bool sameEtaProxyLeadP = false;
            bool samePhiProxyLeadP = false;
            if (proxyCache.leadP.isValid) {
              // bit-by-bit checks of correspondence (asked for no-lint on these "paranoid" checks):
              // (inexpensive enough that we can check this at production time)
              samePtProxyLeadP = (mixLeadP.proxy.pt == leadP.pt);    // NOLINT(clang-diagnostic-float-equal)
              sameEtaProxyLeadP = (mixLeadP.proxy.eta == leadP.eta); // NOLINT(clang-diagnostic-float-equal)
              samePhiProxyLeadP = (mixLeadP.proxy.phi == leadP.phi); // NOLINT(clang-diagnostic-float-equal)
              if (samePtProxyLeadP && sameEtaProxyLeadP && samePhiProxyLeadP) { // Astronomically small chance, even with the birthday paradox
                LOG(fatal) << "EventMixing: borrowed leadP is bit-identical to the target's own in pt, eta and phi. Target collision "
                           << collId << ", source " << mixLeadP.sourceCollisionId << ", pt " << leadP.pt;
              }
            }

            if (doMixingQA) {
              // Filled hit or miss, so the zero bin is the miss rate seen from the pool's side (once per collision, not per resampling pass):
              if (idxResampling == 0) {
                const int nCandidates = leadPByCollision.candidateCountVec[collSlot]; // Zero-initialised, so no existence check needed
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/LeadP/hMixedEventCandidates"), nCandidates);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/LeadP/h2dMixedCandidatesVsPt"), nCandidates, leadP.pt);
                if (doResamplingPassQA)
                  histos.fill(HIST("EventMixingQA/CollLoopOutcome/LeadP/hMixedEventDistinctSources"), distinctSourcesOf(leadPByCollision, collSlot));
              }

              if (proxyCache.leadP.isValid) {
                // Only fill the comparison histograms on an actual hit (source vs. this collision's own):
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/LeadP/hMixedEventOutcome"), 1);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/LeadP/h2dMixedEventOutcomeVsTargetZVtx"), 1, collisionPVz);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/LeadP/h2dMixedEventOutcomeVsTargetCentrality"), 1, centrality);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/LeadP/h2dMixedEventOutcomeVsTargetProxyPt"), 1, leadP.pt);

                histos.fill(HIST("EventMixingQA/AnglrCorrltns/LeadP/h2dMixedProxyEtaVsProxyEta"), proxyCache.leadP.eta, leadP.eta);
                histos.fill(HIST("EventMixingQA/AnglrCorrltns/LeadP/h2dMixedProxyPhiVsProxyPhi"), proxyCache.leadP.phi, leadP.phi);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/LeadP/hMixedEventDeltaPt"), proxyCache.leadP.pt - leadP.pt);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/LeadP/hMixedEventDeltaZvtx"), mixLeadP.zvtx - collisionPVz);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/LeadP/hMixedEventDeltaCent"), mixLeadP.centrality - centrality);

                // QAing the binning policy itself to verify if any wrong proxies are being borrowed:
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/LeadP/h2dMixedEventDeltaPtVsTargetPt"), proxyCache.leadP.pt - leadP.pt, leadP.pt);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/LeadP/h2dMixedEventDeltaZvtxVsTargetZvtx"), mixLeadP.zvtx - collisionPVz, collisionPVz);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/LeadP/h2dMixedEventDeltaCentVsTargetCent"), mixLeadP.centrality - centrality, centrality);

                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/LeadP/hMixedEventDeltaEta"), proxyCache.leadP.eta - leadP.eta);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/LeadP/hMixedEventDeltaPhi"), wrapToPiFast(proxyCache.leadP.phi - leadP.phi));

                // 2D Index Vs Delta Mixed-Variable QAs:
                const float deltaIndexLeadP = static_cast<float>(collId - mixLeadP.sourceCollisionId);
                histos.fill(HIST("EventMixingQA/IndexQA/LeadP/h2dMixedEventDeltaPtVsDeltaIndex"), deltaIndexLeadP, mixLeadP.proxy.pt - leadP.pt);
                histos.fill(HIST("EventMixingQA/IndexQA/LeadP/h2dMixedEventDeltaZvtxVsDeltaIndex"), deltaIndexLeadP, mixLeadP.zvtx - collisionPVz);
                histos.fill(HIST("EventMixingQA/IndexQA/LeadP/h2dMixedEventDeltaCentVsDeltaIndex"), deltaIndexLeadP, mixLeadP.centrality - centrality);

                // Computing final needed bool, which can be declared as a const:
                const bool sameZvtxLeadP = (mixLeadP.zvtx == collisionPVz); // NOLINT(clang-diagnostic-float-equal)

                if (sameZvtxLeadP && samePtProxyLeadP) // Flagged as "alarm" level
                  LOG(alarm) << "EventMixing: source and target share Zvtx and leadP pt bit-for-bit -- duplicated collision row? Target "
                             << collId << ", source " << mixLeadP.sourceCollisionId << ", Zvtx " << collisionPVz;

                // Single-variable coincidences are possible by chance, so they are counted instead:
                if (samePtProxyLeadP)
                  histos.fill(HIST("EventMixingQA/IdentityChecks/LeadP/hMixedEventIdentityFlags"), 0);
                if (sameEtaProxyLeadP)
                  histos.fill(HIST("EventMixingQA/IdentityChecks/LeadP/hMixedEventIdentityFlags"), 1);
                if (samePhiProxyLeadP)
                  histos.fill(HIST("EventMixingQA/IdentityChecks/LeadP/hMixedEventIdentityFlags"), 2);
                if (sameZvtxLeadP)
                  histos.fill(HIST("EventMixingQA/IdentityChecks/LeadP/hMixedEventIdentityFlags"), 3);

                // Possible shared-track case in event splitting, done in a log scale:
                const float angularSepLeadP = std::hypot(mixLeadP.proxy.eta - leadP.eta, wrapToPiFast(mixLeadP.proxy.phi - leadP.phi));
                histos.fill(HIST("EventMixingQA/IdentityChecks/LeadP/hMixedEventLogAngularSep"), angularSepLeadP > 0.f ? std::log10(angularSepLeadP) : -7.f);
              } else {
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/LeadP/hMixedEventOutcome"), 0);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/LeadP/h2dMixedEventOutcomeVsTargetZVtx"), 0, collisionPVz);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/LeadP/h2dMixedEventOutcomeVsTargetCentrality"), 0, centrality);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/LeadP/h2dMixedEventOutcomeVsTargetProxyPt"), 0, leadP.pt);
              }
            }
          }

          // Apply distortion logic:
          // (modifies (leadP, leadPUnitVec) as if the modified proxy was the actual proxy of this event)
          applyProxyDistortion(leadP, leadPUnitVec, minLeadParticlePt, fakePolSwitches.maxLeadPProxyEta, proxyCache.leadP, etaLeadPDist, phiLeadPDist);

          // Fill distorted-proxy QA histograms (i.e., the actually used proxy).
          // A borrowed-proxy miss and a failed pT gate will skip the fill.
          if (leadP.isValid) {
            histos.fill(HIST("KinematicsQA/LeadP/hEta"), leadP.eta);
            histos.fill(HIST("KinematicsQA/LeadP/hPhi"), leadP.phi);
            histos.fill(HIST("KinematicsQA/LeadP/hPt"), leadP.pt);

            histos.fill(HIST("KinematicsQA/LeadP/h2dEtaVsPhi"), leadP.eta, leadP.phi);
            histos.fill(HIST("KinematicsQA/LeadP/h2dEtaVsPVz"), leadP.eta, collisionPVz);
          }
        }

        // 3) Fetching leading jet and subleading jet -- Resolved once per collision in the leadJetByCollision/subJetByCollision pre-pass:
        // (Local copies, as applyProxyDistortion may change them for this collision only)
        ProxyKinematics leadJet = leadJetByCollision.proxyKinematicsVec[collSlot];
        ProxyKinematics subJet = subJetByCollision.proxyKinematicsVec[collSlot];

        // Build jet vectors (only when the corresponding jet exists):
        // Dummy initialisations are safe: all jet-dependent fills are gated on leadJet.isValid / subJet.isValid.
        // Useful scalars for PrimeJet coordinates:
        float leadJetZ = 0.;
        float inverseJetTransverse = 0.;
        XYZVector leadJetUnitVec(1., 0., 0.); // dummy (overwritten below)
        if (leadJet.isValid) {
          // Rebuild the direction from the cached eta/phi (cheaper than calling jetPx() internal getters and then normalizing with .Unit()):
          const double inverseCoshEta = 1.0 / std::cosh(leadJet.eta);
          leadJetUnitVec = XYZVector(std::cos(leadJet.phi) * inverseCoshEta, std::sin(leadJet.phi) * inverseCoshEta, std::tanh(leadJet.eta));

          // Apply distortion logic:
          if (fakePolSwitches.doMixedEventProxies) {
            // Get this collision's leading-jet mixing partner (if any) from the LUT built above:
            const MixedProxyInfo mixLeadJet = rebuildBorrowedProxyFromCache(leadJetByCollision, mixingAxesByCollision, collisionIndexBase, collSlot, idxResampling);
            proxyCache.leadJet = mixLeadJet.proxy;
            if (proxyCache.leadJet.isValid) {
              // bit-by-bit checks of correspondence:
              const bool samePtProxyLeadJet = (mixLeadJet.proxy.pt == leadJet.pt);    // NOLINT(clang-diagnostic-float-equal)
              const bool sameEtaProxyLeadJet = (mixLeadJet.proxy.eta == leadJet.eta); // NOLINT(clang-diagnostic-float-equal)
              const bool samePhiProxyLeadJet = (mixLeadJet.proxy.phi == leadJet.phi); // NOLINT(clang-diagnostic-float-equal)
              if (samePtProxyLeadJet && sameEtaProxyLeadJet && samePhiProxyLeadJet) { // Astronomically small chance, even with the birthday paradox
                LOG(fatal) << "EventMixing: borrowed LeadJet is bit-identical to the target's own in pt, eta and phi. Target collision "
                           << collId << ", source " << mixLeadJet.sourceCollisionId << ", pt " << leadJet.pt;
              }
            }

            if (doMixingQA) {
              if (idxResampling == 0) {
                const int nCandidates = leadJetByCollision.candidateCountVec[collSlot];
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/LeadJet/hMixedEventCandidates"), nCandidates);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/LeadJet/h2dMixedCandidatesVsPt"), nCandidates, leadJet.pt);
                if (doResamplingPassQA)
                  histos.fill(HIST("EventMixingQA/CollLoopOutcome/LeadJet/hMixedEventDistinctSources"), distinctSourcesOf(leadJetByCollision, collSlot));
              }

              if (proxyCache.leadJet.isValid) {
                // Only fill the comparison histograms on an actual hit (source vs. this collision's own):
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/LeadJet/hMixedEventOutcome"), 1);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/LeadJet/h2dMixedEventOutcomeVsTargetZVtx"), 1, collisionPVz);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/LeadJet/h2dMixedEventOutcomeVsTargetCentrality"), 1, centrality);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/LeadJet/h2dMixedEventOutcomeVsTargetProxyPt"), 1, leadJet.pt);

                histos.fill(HIST("EventMixingQA/AnglrCorrltns/LeadJet/h2dMixedProxyEtaVsProxyEta"), proxyCache.leadJet.eta, leadJet.eta);
                histos.fill(HIST("EventMixingQA/AnglrCorrltns/LeadJet/h2dMixedProxyPhiVsProxyPhi"), proxyCache.leadJet.phi, leadJet.phi);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/LeadJet/hMixedEventDeltaPt"), proxyCache.leadJet.pt - leadJet.pt);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/LeadJet/hMixedEventDeltaZvtx"), mixLeadJet.zvtx - collisionPVz);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/LeadJet/hMixedEventDeltaCent"), mixLeadJet.centrality - centrality);

                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/LeadJet/hMixedEventDeltaEta"), proxyCache.leadJet.eta - leadJet.eta);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/LeadJet/hMixedEventDeltaPhi"), wrapToPiFast(proxyCache.leadJet.phi - leadJet.phi));
              } else {
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/LeadJet/hMixedEventOutcome"), 0);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/LeadJet/h2dMixedEventOutcomeVsTargetZVtx"), 0, collisionPVz);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/LeadJet/h2dMixedEventOutcomeVsTargetCentrality"), 0, centrality);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/LeadJet/h2dMixedEventOutcomeVsTargetProxyPt"), 0, leadJet.pt);
              }
            }
          }
          applyProxyDistortion(leadJet, leadJetUnitVec, minLeadJetPt, fakePolSwitches.maxJetProxyEta, proxyCache.leadJet, etaLeadPDist, phiLeadPDist);

          // Fill distorted-proxy QA histograms:
          // Do not gate on the post-distortion leadJet.isValid (a pT cut) value here!
          if (leadJet.isValid) {
            histos.fill(HIST("KinematicsQA/LeadJet/hEta"), leadJet.eta);
            histos.fill(HIST("KinematicsQA/LeadJet/hPhi"), leadJet.phi);
            histos.fill(HIST("KinematicsQA/LeadJet/hPt"), leadJet.pt);

            histos.fill(HIST("KinematicsQA/LeadJet/h2dEtaVsPhi"), leadJet.eta, leadJet.phi);
            histos.fill(HIST("KinematicsQA/LeadJet/h2dEtaVsPVz"), leadJet.eta, collisionPVz);

            // PrimeJet coordinate system variables:
            // (Calculated once per collision, after applyProxyDistortion, using a Gram-Schmidt-like orthonormalisation)
            // \hat x_Jet = (\hat z - t_z \hat t)/t_T, \hat y_Jet = (\hat t \times \hat z)/t_T
            leadJetZ = leadJetUnitVec.Z(); // \hat z_Jet = \hat t
            const float jetFrameTt = std::sqrt(1.0f - leadJetZ * leadJetZ); // t_T = sqrt(1 - t_z^2) = 1/cosh(eta_jet)
            inverseJetTransverse = (jetFrameTt > 1e-4f) ? (1.0f / jetFrameTt) : 0.0f; // Paranoid guard with the pseudorapidity cuts, but safer nonetheless
          }
        }

        XYZVector subJetUnitVec(1., 0., 0.);
        if (subJet.isValid) {
          const double inverseCoshEtaSub = 1.0 / std::cosh(subJet.eta);
          subJetUnitVec = XYZVector(std::cos(subJet.phi) * inverseCoshEtaSub, std::sin(subJet.phi) * inverseCoshEtaSub, std::tanh(subJet.eta));

          // Apply distortion logic:
          if (fakePolSwitches.doMixedEventProxies) {
            // Get this collision's subleading-jet mixing partner (if any) from the LUT built above:
            const MixedProxyInfo mixSubJet = rebuildBorrowedProxyFromCache(subJetByCollision, mixingAxesByCollision, collisionIndexBase, collSlot, idxResampling);
            proxyCache.subJet = mixSubJet.proxy;
            if (proxyCache.subJet.isValid) {
              // bit-by-bit checks of correspondence:
              const bool samePtProxySubJet = (mixSubJet.proxy.pt == subJet.pt);    // NOLINT(clang-diagnostic-float-equal)
              const bool sameEtaProxySubJet = (mixSubJet.proxy.eta == subJet.eta); // NOLINT(clang-diagnostic-float-equal)
              const bool samePhiProxySubJet = (mixSubJet.proxy.phi == subJet.phi); // NOLINT(clang-diagnostic-float-equal)
              if (samePtProxySubJet && sameEtaProxySubJet && samePhiProxySubJet) { // Astronomically small chance, even with the birthday paradox
                LOG(fatal) << "EventMixing: borrowed SubJet is bit-identical to the target's own in pt, eta and phi. Target collision "
                           << collId << ", source " << mixSubJet.sourceCollisionId << ", pt " << subJet.pt;
              }
            }

            if (doMixingQA) {
              if (idxResampling == 0) {
                const int nCandidates = subJetByCollision.candidateCountVec[collSlot];
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/SubJet/hMixedEventCandidates"), nCandidates);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/SubJet/h2dMixedCandidatesVsPt"), nCandidates, subJet.pt);
                if (doResamplingPassQA)
                  histos.fill(HIST("EventMixingQA/CollLoopOutcome/SubJet/hMixedEventDistinctSources"), distinctSourcesOf(subJetByCollision, collSlot));
              }

              if (proxyCache.subJet.isValid) {
                // Only fill the comparison histograms on an actual hit (source vs. this collision's own):
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/SubJet/hMixedEventOutcome"), 1);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/SubJet/h2dMixedEventOutcomeVsTargetZVtx"), 1, collisionPVz);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/SubJet/h2dMixedEventOutcomeVsTargetCentrality"), 1, centrality);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/SubJet/h2dMixedEventOutcomeVsTargetProxyPt"), 1, subJet.pt);

                histos.fill(HIST("EventMixingQA/AnglrCorrltns/SubJet/h2dMixedProxyEtaVsProxyEta"), proxyCache.subJet.eta, subJet.eta);
                histos.fill(HIST("EventMixingQA/AnglrCorrltns/SubJet/h2dMixedProxyPhiVsProxyPhi"), proxyCache.subJet.phi, subJet.phi);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/SubJet/hMixedEventDeltaPt"), proxyCache.subJet.pt - subJet.pt);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/SubJet/hMixedEventDeltaZvtx"), mixSubJet.zvtx - collisionPVz);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/SubJet/hMixedEventDeltaCent"), mixSubJet.centrality - centrality);

                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/SubJet/hMixedEventDeltaEta"), proxyCache.subJet.eta - subJet.eta);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/SubJet/hMixedEventDeltaPhi"), wrapToPiFast(proxyCache.subJet.phi - subJet.phi));
              } else {
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/SubJet/hMixedEventOutcome"), 0);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/SubJet/h2dMixedEventOutcomeVsTargetZVtx"), 0, collisionPVz);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/SubJet/h2dMixedEventOutcomeVsTargetCentrality"), 0, centrality);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/SubJet/h2dMixedEventOutcomeVsTargetProxyPt"), 0, subJet.pt);
              }
            }
          }
          applyProxyDistortion(subJet, subJetUnitVec, minSubLeadJetPt, fakePolSwitches.maxJetProxyEta, proxyCache.subJet, etaLeadPDist, phiLeadPDist);

          // Fill distorted-proxy QA histograms:
          // Do not gate on the post-distortion subJet.isValid (a pT cut) value here!
          if (subJet.isValid) {
            histos.fill(HIST("KinematicsQA/SubJet/hEta"), subJet.eta);
            histos.fill(HIST("KinematicsQA/SubJet/hPhi"), subJet.phi);
            histos.fill(HIST("KinematicsQA/SubJet/hPt"), subJet.pt);

            histos.fill(HIST("KinematicsQA/SubJet/h2dEtaVsPhi"), subJet.eta, subJet.phi);
            histos.fill(HIST("KinematicsQA/SubJet/h2dEtaVsPVz"), subJet.eta, collisionPVz);
          }
        }

        // forcePreviousJet: hand this collision's own proxies to the next collision.
        if (fakePolSwitches.forcePreviousJet) {
          prevJetCache = SetOfProxies{beforeDistLeadP, leadJetByCollision.proxyKinematicsVec[collSlot], subJetByCollision.proxyKinematicsVec[collSlot]};
        }

        // (jet eta cuts only meaningful when the jet actually exists)
        const bool kinematicLeadJetCheck = leadJet.isValid && (std::abs(leadJet.eta) < 0.5);
        const bool kinematicSubJetCheck = subJet.isValid && (std::abs(subJet.eta) < 0.5);
        const bool kinematicLeadPCheck = leadP.isValid && (std::abs(leadP.eta) < 0.5);

        // Quick bools that are useful for detector asymmetry QA:
        const bool leadJetEtaPos = leadJet.isValid && (leadJet.eta >= 0.); // Only perform >= check if has validJet
        const bool subJetEtaPos = subJet.isValid && (subJet.eta >= 0.);
        const bool leadPEtaPos = leadP.isValid && (leadP.eta >= 0.);

        // Stricter QA version of the bools -- Jets have a radius that makes it possible eta_{jet} > 0, yet half its tracks are in eta < 0
        // (This does not apply to leading particles, obviously. They have no substructure in eta)
        const bool leadJetEtaStrict = leadJet.isValid && (std::abs(leadJet.eta) >= jetR);
        const bool subJetEtaStrict = subJet.isValid && (std::abs(subJet.eta) >= jetR);
        // If one was to define bools for each side of the detector (not needed in the current if-else structure on TProfile fills)
        // const bool leadJetEtaStrictPos = leadJetEtaPos && leadJetEtaStrict;
        // const bool leadJetEtaStrictNeg = !leadJetEtaPos && leadJetEtaStrict;
        // const bool subJetEtaStrictPos = subJetEtaPos && subJetEtaStrict;
        // const bool subJetEtaStrictNeg = !subJetEtaPos && subJetEtaStrict;

        // Initialize delta method accumulators (reset for new collision):
        for (auto const& tracker : {&trackRing, &trackRingKinCuts, &trackJetKinCuts, &trackJetLambdaKinCuts})
          tracker->reset();
        for (auto const& v0 : v0sInColl) {
          const bool isLambda = v0.isLambda(); // true: is a Lambda. false: is an antiLambda.
          // For now, removing the ambiguous candidates from the analysis. New datamodel does NOT save ambiguous candidates.
          // (From Podolanski-Armenteros plots, the population of ambiguous is ~3.8% without TOF, and without
          //  competing mass rejection. From those, ~99% seem to be K0s, so no real gain in considering the
          //  ambiguous candidates in the analysis)
          // const bool isAntiLambda = v0.isAntiLambda(); // No longer used!
          // if (isLambda && isAntiLambda) continue;
 
          // Species gate:
          if ((isLambda && !analyseLambda) || (!isLambda && !analyseAntiLambda))
            continue;
          // Additional analysis-level cuts before caching variables:
          if (analysisLevelCuts.doAnalysisLevelCuts && !isV0Accepted(v0))
            continue;
 
          const float v0pt = v0.v0Pt();
          const float v0eta = v0.v0Eta();
          const float v0phi = v0.v0Phi();
          const float v0LambdaLikeMass = v0.massV0();
          float protonLikePt = 0;
          float protonLikeEta = 0;
          float protonLikePhi = 0;
          float protonLikeDCADauToPV = 0;
          float pionLikePt = 0; // Pion variables just for QA
          float pionLikeEta = 0;
          float pionLikePhi = 0;
          float pionLikeDCADauToPV = 0;
          const float dcaDau = v0.dcaV0Daughters();
          if (isLambda) {
            protonLikePt = v0.posPt();
            protonLikeEta = v0.posEta();
            protonLikePhi = v0.posPhi();
            protonLikeDCADauToPV = v0.dcaPosToPV();

            pionLikePt = v0.negPt();
            pionLikeEta = v0.negEta();
            pionLikePhi = v0.negPhi();
            pionLikeDCADauToPV = v0.dcaNegToPV();
          } else { // Guaranteed to be an antiLambda candidate, not an ambiguous candidate
            protonLikePt = v0.negPt();
            protonLikeEta = v0.negEta();
            protonLikePhi = v0.negPhi();
            protonLikeDCADauToPV = v0.dcaNegToPV();

            pionLikePt = v0.posPt();
            pionLikeEta = v0.posEta();
            pionLikePhi = v0.posPhi();
            pionLikeDCADauToPV = v0.dcaPosToPV();
          }

          // Kinematics QA of V0 and daughters at derived data level, post-selection:
          histos.fill(HIST("KinematicsQA/V0/hV0Phi"), v0phi);
          histos.fill(HIST("KinematicsQA/V0/hV0Eta"), v0eta);
          histos.fill(HIST("KinematicsQA/V0/hV0Pt"), v0pt);
          histos.fill(HIST("KinematicsQA/V0/h2dV0PhiVsEta"), v0phi, v0eta);
          if (leadJet.isValid) { // Using post-distortion/mixing variables for the jets
            histos.fill(HIST("KinematicsQA/V0/h2dV0PhiVsLeadJetPhi"), v0phi, leadJet.phi);
            histos.fill(HIST("KinematicsQA/V0/h2dV0EtaVsLeadJetEta"), v0eta, leadJet.eta);
          }
          // Proton-like daughter:
          histos.fill(HIST("KinematicsQA/V0dau/hPrPhi"), protonLikePhi);
          histos.fill(HIST("KinematicsQA/V0dau/hPrEta"), protonLikeEta);
          histos.fill(HIST("KinematicsQA/V0dau/hPrPt"), protonLikePt);
          histos.fill(HIST("KinematicsQA/V0dau/h2dPrPhiVsEta"), protonLikePhi, protonLikeEta);
          // Pion-like daughter:
          histos.fill(HIST("KinematicsQA/V0dau/hPiPhi"), pionLikePhi);
          histos.fill(HIST("KinematicsQA/V0dau/hPiEta"), pionLikeEta);
          histos.fill(HIST("KinematicsQA/V0dau/hPiPt"), pionLikePt);
          histos.fill(HIST("KinematicsQA/V0dau/h2dPiPhiVsEta"), pionLikePhi, pionLikeEta);
          // V0 vs proton-like daughter:
          histos.fill(HIST("KinematicsQA/V0dau/h2dV0PhiVsPrPhi"), v0phi, protonLikePhi);
          histos.fill(HIST("KinematicsQA/V0dau/h2dV0EtaVsPrEta"), v0eta, protonLikeEta);
          // V0 vs pion-like daughter:
          histos.fill(HIST("KinematicsQA/V0dau/h2dV0PhiVsPiPhi"), v0phi, pionLikePhi);
          histos.fill(HIST("KinematicsQA/V0dau/h2dV0EtaVsPiEta"), v0eta, pionLikeEta);
          // Proton-like vs pion-like daughters:
          histos.fill(HIST("KinematicsQA/V0dau/h2dPrPhiVsPiPhi"), protonLikePhi, pionLikePhi);
          histos.fill(HIST("KinematicsQA/V0dau/h2dPrEtaVsPiEta"), protonLikeEta, pionLikeEta);

          PtEtaPhiMVector lambdaLike4Vec(v0pt, v0eta, v0phi, v0LambdaLikeMass);
          PtEtaPhiMVector protonLike4Vec(protonLikePt, protonLikeEta, protonLikePhi, ProtonMass);
          const float lambdaRapidity = lambdaLike4Vec.Rapidity();                                // For further kinematic selections
          // const int v0InMassPeak = (v0LambdaLikeMass >= 1.11014 && v0LambdaLikeMass <= 1.12061); // Very naive estimator, \pm 3\sigma. Based on signal extractions from outside this code
          // Naive estimator based on signal extractions from outside this code:
          const bool v0InMassPeak = (v0LambdaLikeMass <= (LambdaMass + PeakWindowNSigma*LambdaMassSigma) && v0LambdaLikeMass >= (LambdaMass - PeakWindowNSigma*LambdaMassSigma));
          const bool v0InMassWindow = (v0LambdaLikeMass >= (LambdaMass - SidebandOuterNSigma*LambdaMassSigma) && v0LambdaLikeMass < (LambdaMass - SidebandInnerNSigma*LambdaMassSigma)) ||
                                      (v0LambdaLikeMass >= (LambdaMass + SidebandInnerNSigma*LambdaMassSigma) && v0LambdaLikeMass < (LambdaMass + SidebandOuterNSigma*LambdaMassSigma));
          // The same two flags as one fill value for the *VsMassRegion profiles: 1.5 peak, 0.5 sideband, -1 neither (underflow)
          const float massRegion = v0InMassPeak ? 1.5f : (v0InMassWindow ? 0.5f : -1.f);

          // Inexpensive estimates of signal extraction effects on the observable:
          if (excludeOutOfPeakQA && !v0InMassPeak)
            continue;
          else if (excludeInPeakQA && !v0InMassWindow)
            continue;

          // Boosting proton into lambda frame:
          XYZVector beta = lambdaLike4Vec.BoostToCM(); // Boost trivector that goes from laboratory frame to Lambda's rest frame (convenient new function, different from TLorentzVector's BoostVector())
          auto protonLike4VecStar = ROOT::Math::VectorUtil::boost(protonLike4Vec, beta);

          // Getting unit vectors and 3-components:
          XYZVector lambdaLike3Vec = lambdaLike4Vec.Vect();
          auto lambdaLikeUnit3Vec = lambdaLike3Vec.Unit();
          const float lambdaZ = lambdaLikeUnit3Vec.Z(); // cos(theta_Lambda) = tanh(eta_Lambda)
          XYZVector protonLikeStarUnit3Vec = protonLike4VecStar.Vect().Unit();

          // Lab-frame Lambda momentum components -- Not for polarization, but for actual momenta plotting in XY and ZX planes:
          // (Used for the (px,py) and (pz,px) polarization vector-field / ring 2D profiles)
          const float v0px = lambdaLike3Vec.X();
          const float v0py = lambdaLike3Vec.Y();
          const float v0pz = lambdaLike3Vec.Z();

          // Calculating fake polarization ("negative helicity problem") estimator:
          // (this estimator is calculated outside of any gate, as it does not depend on jet proxy used)
          float cosFakePol = protonLikeStarUnit3Vec.Dot(lambdaLikeUnit3Vec);

          // Calculating the azimuthal angle between the Lambda and the proton:
          // (Phi is defined in (-PI,PI] in ROOT::Math::Cartesian3D, thus kept the wrapping)
          float deltaPhiLambdaProtonStar = wrapToPiFast(lambdaLikeUnit3Vec.Phi() - protonLikeStarUnit3Vec.Phi());
          // Rewriting the proton star angle in a [0,2PI) interval (axis conventions), as ROOT::Math::Cartesian3D defines it in (-PI,PI]:
          const float protonLikeStarPhiWrap = RecoDecay::constrainAngle(protonLikeStarUnit3Vec.Phi(), 0.f);

          // phi*: azimuth of the proton* around the Lambda direction, measured from z_hat x p_Lambda_hat.
          // With e_z = p_Lambda_hat, e_x = z_hat x p_Lambda_hat and e_y = e_z x e_x = z_hat - L_z * p_Lambda_hat:
          //   pX = p_D_hat^* \cdot e_x = p_D,y^* * L_x - p_D,x^* * L_y
          //   pY = p_D_hat^* \cdot e_y = p_D,z^* - L_z * (p_D_hat^* \cdot p_Lambda_hat)
          // e_x and e_y are left unnormalized on purpose, because atan2 would not see their sin(theta_Lambda) norm (saves up a normalization)
          float pX = protonLikeStarUnit3Vec.Y() * lambdaLikeUnit3Vec.X() - protonLikeStarUnit3Vec.X() * lambdaLikeUnit3Vec.Y();
          float pY = protonLikeStarUnit3Vec.Z() - lambdaLikeUnit3Vec.Z() * cosFakePol; // (Reusing cosFakePol calculated earlier!)
          float phiStar = std::atan2(pY, pX);                                          // This will give an output from -PI to PI

          // Ring prefactor: same as polPrefactor, but forcePolSignQA inverts it for antiLambdas only
          const float polPrefactor = isLambda ? PolPrefactorLambda : PolPrefactorAntiLambda;
          const float ringPrefactor = (fakePolSwitches.forcePolSignQA && !isLambda) ? -polPrefactor : polPrefactor;
          const float v0p = longitudinalRingMode ? lambdaLike3Vec.R() : 0.f; // |p_Lambda|, only used for the R_z bearing chi
          // Longitudinal rings: p* projected on the beam axis (R_z) or on the beam axis made transverse to the Lambda (R_z^T),
          // which removes the helicity component (p* \cdot p_Lambda) \hat p_Lambda,z. That component is kept apart for the diagnostics.
          const float protonStarHelicityBeamComponent = cosFakePol * lambdaLikeUnit3Vec.Z(); // Equals (p* \cdot p_Lambda) \hat p_Lambda,z
          const float protonStarRingBeamComponent = transverseBeamAxisMode ? protonLikeStarUnit3Vec.Z() - protonStarHelicityBeamComponent
                                                                           : protonLikeStarUnit3Vec.Z();
          // TODO: Properly gate these to avoid unneeded computations whilst in the full 3D ring mode (i.e., non useRingZ nor useRingZT)

          // Calculating polarization observables (in the Lambda frame, because that is easier -- does not require boosts):
          // To be precise, not the polarization itself, but a part of the summand in P^*_Lambda = (3/\alpha_Lambda) * <p^*_{proton}>
          const float polStarX = polPrefactor * protonLikeStarUnit3Vec.X();
          const float polStarY = polPrefactor * protonLikeStarUnit3Vec.Y();
          const float polStarZ = polPrefactor * protonLikeStarUnit3Vec.Z();

          // Calculating rotated coordinate systems:
          // AEE-frame's relevant polarization:
          const float protonStarPt = protonLikeStarUnit3Vec.Rho(); // TODO: check if this makes sense. Should it be the un-normalized version instead?

          // Guarded inverse values for calculating rotations, similar to the leadJetZ guard below:
          const float invProtonStarPt = (protonStarPt > 1e-6f) ? (1.0f / protonStarPt) : 0.0f;
          const float invV0pt = (v0pt > 1e-6f) ? (1.0f / v0pt) : 0.0f;

          // Rotating into "Aee frame" (rotated by -phi_proton*):
          // p_{x,AEE} = pT^Lambda cos(phi_Lambda - phi_p*),  p_{y,AEE} = pT^Lambda sin(phi_Lambda - phi_p*)
          const float v0pxAee = (v0px * protonLikeStarUnit3Vec.X() + v0py * protonLikeStarUnit3Vec.Y()) * invProtonStarPt;
          const float v0pyAee = (v0py * protonLikeStarUnit3Vec.X() - v0px * protonLikeStarUnit3Vec.Y()) * invProtonStarPt;

          // Aee-frame polarization:
          // (By construction (|P*_T|, 0, P*_z), and only the magnitude varies)
          const float polStarTAee = polPrefactor * protonStarPt;

          // PrimeV0 frame: -- polStarYPrimeV0 is the ring observable computed with the beam as the jet proxy 
          // (a control that carries no jet-correlated ring signal, but the full AEE bias and possible \hat z vortices)
          const float polStarXPrimeV0 = polPrefactor * (protonLikeStarUnit3Vec.X() * v0px + protonLikeStarUnit3Vec.Y() * v0py) * invV0pt;
          const float polStarYPrimeV0 = polPrefactor * (protonLikeStarUnit3Vec.Y() * v0px - protonLikeStarUnit3Vec.X() * v0py) * invV0pt;

          if (qaSwitches.doFakePolDiagnosticsQA) {
            // Another reconstruction efficiency measure:
            // (Formula is: p_{Lambda} \cross p_{Daughter}^{*} \cdot B, and B points in Z)
            if (analyseMagField) {
              auto crossGeom = lambdaLike3Vec.Cross(protonLikeStarUnit3Vec);
              const bool positiveGeom = crossGeom.Z() * magField > 0;

              if (isLambda && positiveGeom)
                histos.fill(HIST("HelicityEfficiencyQA/hLambdaMassDecayGeomRight"), v0LambdaLikeMass);
              else if (isLambda && !positiveGeom)
                histos.fill(HIST("HelicityEfficiencyQA/hLambdaMassDecayGeomLeft"), v0LambdaLikeMass);
              else if (!isLambda && positiveGeom)
                histos.fill(HIST("HelicityEfficiencyQA/hAntiLambdaMassDecayGeomRight"), v0LambdaLikeMass);
              else
                histos.fill(HIST("HelicityEfficiencyQA/hAntiLambdaMassDecayGeomLeft"), v0LambdaLikeMass);
            }

            // Measuring the AEE effect differentially (azimuthal efficiency effect, which causes different V0 topologies to be enhanced/suppressed):
            if (isLambda)
              histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/h2dLambdaMassVsPhiAEE"), v0LambdaLikeMass, deltaPhiLambdaProtonStar);
            else
              histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/h2dAntiLambdaMassVsPhiAEE"), v0LambdaLikeMass, deltaPhiLambdaProtonStar);
            // AEE and HEE correlation:
            histos.fill(HIST("HelicityEfficiencyQA/h2dFakePolCounts_CosThetaVsPhiStar"), cosFakePol, phiStar);
          } // end doFakePolDiagnosticsQA (per-V0 AEE/HEE)

          // Useful kinematic bools:
          const bool lambdaEtaPos = v0eta >= 0.;
          const bool pTLambdaCheck = v0pt > 0.5 && v0pt < 1.5;
          const bool rapidityLambdaCheck = std::abs(lambdaRapidity) < 0.5;
          const bool kinematicLambdaCheck = pTLambdaCheck && rapidityLambdaCheck;

          // Filling ProtonStar kinematic QAs:
          histos.fill(HIST("KinematicsQA/ProtonStar/hPrStarPhi"), protonLikeStarPhiWrap);
          histos.fill(HIST("KinematicsQA/ProtonStar/hPrStarEta"), protonLikeStarUnit3Vec.Eta());
          histos.fill(HIST("KinematicsQA/ProtonStar/hPrStarPt"), protonLike4VecStar.Rho()); // Need to use non-normalized vector here
          histos.fill(HIST("KinematicsQA/ProtonStar/h2dPrStarPhiVsEta"), protonLikeStarPhiWrap, protonLikeStarUnit3Vec.Eta());
          // V0 vs protonStar-like daughter:
          histos.fill(HIST("KinematicsQA/ProtonStar/h2dV0PhiVsPrStarPhi"), v0phi, protonLikeStarPhiWrap);
          histos.fill(HIST("KinematicsQA/ProtonStar/h2dV0EtaVsPrStarEta"), v0eta, protonLikeStarUnit3Vec.Eta());

          // Per-V0 inputs of the ring, shared by every proxy:
          const V0RingInputs v0RingInputs{lambdaLike3Vec, lambdaLikeUnit3Vec, protonLikeStarUnit3Vec, v0phi, v0p, v0px, v0py,
                                          ringPrefactor, protonStarRingBeamComponent, protonStarHelicityBeamComponent};

          ////////////////////////////////////////////
          // Ring observable: one computeProxyRing() call per proxy
          // Only computed when the proxy is valid. A missing proxy keeps the ProxyRingCache defaults.
          // (each proxy needs its own minimum pT, see the isValid flags)
          ////////////////////////////////////////////
          const ProxyRingCache ringLeadP = leadP.isValid ? computeProxyRing(leadPUnitVec, leadP.phi, v0RingInputs) : ProxyRingCache{};
          const ProxyRingCache ringLeadJet = leadJet.isValid ? computeProxyRing(leadJetUnitVec, leadJet.phi, v0RingInputs) : ProxyRingCache{};
          const ProxyRingCache ringSubJet = subJet.isValid ? computeProxyRing(subJetUnitVec, subJet.phi, v0RingInputs) : ProxyRingCache{};

          // Leading-jet-only companions of the ring:
          float deltaEtaLeadJet = 0.;
          float ringObservableOverLeadJetZ = 0.;
          // PrimeJet-frame components:
          float polStarXPrimeJet = 0.;
          float polStarYPrimeJet = 0.;
          float polStarZPrimeJet = 0.;
          float v0pxPrimeJet = 0.;
          float v0pyPrimeJet = 0.;
          float v0pzPrimeJet = 0.;
          if (leadJet.isValid) {
            deltaEtaLeadJet = v0eta - leadJet.eta;

            // PrimeJet-frame components, using the per-collision basis scalars:
            const float protonStarDotLeadJet = protonLikeStarUnit3Vec.Dot(leadJetUnitVec);
            polStarZPrimeJet = polPrefactor * protonStarDotLeadJet; // P_{z'Jet} = p* \cdot \hat t (polarization on jet direction)
            polStarXPrimeJet = polPrefactor * (protonLikeStarUnit3Vec.Z() - leadJetZ * protonStarDotLeadJet) * inverseJetTransverse; // P_{x'Jet} = (p*_z - t_z (p* \cdot \hat t)) / t_T
            polStarYPrimeJet = polPrefactor * (protonLikeStarUnit3Vec.X() * leadJetUnitVec.Y() - protonLikeStarUnit3Vec.Y() * leadJetUnitVec.X()) * inverseJetTransverse; // P_{y'Jet} = (p*_x t_y - p*_y t_x) / t_T

            // The same rotation is applied to the Lambda momentum to plot it in this new system:
            v0pzPrimeJet = lambdaLike3Vec.Dot(leadJetUnitVec);
            v0pxPrimeJet = (v0pz - leadJetZ * v0pzPrimeJet) * inverseJetTransverse;
            v0pyPrimeJet = (v0px * leadJetUnitVec.Y() - v0py * leadJetUnitVec.X()) * inverseJetTransverse;

            // Testing an invariance -- <R>/\hat{t}_z -- Possible source of an artificial (trivial) sign flip in the observable:
            if (std::abs(leadJetZ) > 1e-4)
              ringObservableOverLeadJetZ = ringLeadJet.ringObservable / leadJetZ;
            else
              ringObservableOverLeadJetZ = 0.0; // A simple guard. May not be the best, but works
          }

          float v0phiToFillHists = wrapToPiFast(v0phi); // A short wrap to reuse some predefined axes

          // Longitudinal-ring diagnostics (every proxy's ringObservable already holds R_z or R_z^T here):
          if (longitudinalRingMode) {
            if (leadJet.isValid) {
              histos.fill(HIST("RzDiagnostics/pHelicityTermIntegrated"), 0.5, ringLeadJet.helicityTerm);
              histos.fill(HIST("RzDiagnostics/LeadJet/pHelicityTermVsEtaLambda"), v0eta, ringLeadJet.helicityTerm);
              histos.fill(HIST("RzDiagnostics/LeadJet/p2dHelicityTermVsDeltaPhiVsMass"), ringLeadJet.deltaPhi, v0LambdaLikeMass, ringLeadJet.helicityTerm);
              histos.fill(HIST("RzDiagnostics/LeadJet/pNzVsEtaLambda"), v0eta, ringLeadJet.nz);
              histos.fill(HIST("RzDiagnostics/LeadJet/pNzSqVsEtaLambda"), v0eta, ringLeadJet.nz * ringLeadJet.nz);
              if (v0eta > 0.f)
                histos.fill(HIST("RzDiagnostics/LeadJet/p2dRingVsDeltaPhiVsMassEtaPos"), ringLeadJet.deltaPhi, v0LambdaLikeMass, ringLeadJet.ringObservable);
              else
                histos.fill(HIST("RzDiagnostics/LeadJet/p2dRingVsDeltaPhiVsMassEtaNeg"), ringLeadJet.deltaPhi, v0LambdaLikeMass, ringLeadJet.ringObservable);
              histos.fill(HIST("RzDiagnostics/LeadJet/pRingVsChi"), ringLeadJet.chi, ringLeadJet.ringObservable);
              histos.fill(HIST("RzDiagnostics/LeadJet/p2dRingVsChiVsMass"), ringLeadJet.chi, v0LambdaLikeMass, ringLeadJet.ringObservable);
              histos.fill(HIST("RzDiagnostics/LeadJet/p2dRingVsDeltaPhiVsDeltaEta"), ringLeadJet.deltaPhi, deltaEtaLeadJet, ringLeadJet.ringObservable);
              histos.fill(HIST("RzDiagnostics/pRingPerpIntegrated"), 0.5, ringLeadJet.ringPerp);
              histos.fill(HIST("RzDiagnostics/LeadJet/pRingPerpVsMass"), v0LambdaLikeMass, ringLeadJet.ringPerp);
              histos.fill(HIST("RzDiagnostics/LeadJet/pRingPerpVsPhiAEE"), deltaPhiLambdaProtonStar, ringLeadJet.ringPerp);
              histos.fill(HIST("RzDiagnostics/LeadJet/pRingPerpVsDeltaPhi"), ringLeadJet.deltaPhi, ringLeadJet.ringPerp);
              histos.fill(HIST("RzDiagnostics/LeadJet/pCosSqThetaStarZVsMass"), v0LambdaLikeMass, protonLikeStarUnit3Vec.Z() * protonLikeStarUnit3Vec.Z());
            }
            if (leadP.isValid) {
              histos.fill(HIST("RzDiagnostics/pHelicityTermIntegrated"), 1.5, ringLeadP.helicityTerm);
              histos.fill(HIST("RzDiagnostics/LeadP/pHelicityTermVsEtaLambda"), v0eta, ringLeadP.helicityTerm);
              histos.fill(HIST("RzDiagnostics/LeadP/pNzVsEtaLambda"), v0eta, ringLeadP.nz);
              histos.fill(HIST("RzDiagnostics/LeadP/pNzSqVsEtaLambda"), v0eta, ringLeadP.nz * ringLeadP.nz);
              histos.fill(HIST("RzDiagnostics/LeadP/pRingVsChi"), ringLeadP.chi, ringLeadP.ringObservable);
              histos.fill(HIST("RzDiagnostics/LeadP/p2dRingVsChiVsMass"), ringLeadP.chi, v0LambdaLikeMass, ringLeadP.ringObservable);
              histos.fill(HIST("RzDiagnostics/pRingPerpIntegrated"), 1.5, ringLeadP.ringPerp);
              histos.fill(HIST("RzDiagnostics/LeadP/pRingPerpVsMass"), v0LambdaLikeMass, ringLeadP.ringPerp);
              histos.fill(HIST("RzDiagnostics/LeadP/pRingPerpVsPhiAEE"), deltaPhiLambdaProtonStar, ringLeadP.ringPerp);
              histos.fill(HIST("RzDiagnostics/LeadP/pRingPerpVsDeltaPhi"), ringLeadP.deltaPhi, ringLeadP.ringPerp);
            }
            if (subJet.isValid) {
              histos.fill(HIST("RzDiagnostics/pHelicityTermIntegrated"), 2.5, ringSubJet.helicityTerm);
              histos.fill(HIST("RzDiagnostics/SubJet/pHelicityTermVsEtaLambda"), v0eta, ringSubJet.helicityTerm);
              histos.fill(HIST("RzDiagnostics/SubJet/pNzVsEtaLambda"), v0eta, ringSubJet.nz);
              histos.fill(HIST("RzDiagnostics/SubJet/pNzSqVsEtaLambda"), v0eta, ringSubJet.nz * ringSubJet.nz);
              histos.fill(HIST("RzDiagnostics/SubJet/pRingVsChi"), ringSubJet.chi, ringSubJet.ringObservable);
              histos.fill(HIST("RzDiagnostics/SubJet/p2dRingVsChiVsMass"), ringSubJet.chi, v0LambdaLikeMass, ringSubJet.ringObservable);
              histos.fill(HIST("RzDiagnostics/pRingPerpIntegrated"), 2.5, ringSubJet.ringPerp);
              histos.fill(HIST("RzDiagnostics/SubJet/pRingPerpVsMass"), v0LambdaLikeMass, ringSubJet.ringPerp);
              histos.fill(HIST("RzDiagnostics/SubJet/pRingPerpVsPhiAEE"), deltaPhiLambdaProtonStar, ringSubJet.ringPerp);
              histos.fill(HIST("RzDiagnostics/SubJet/pRingPerpVsDeltaPhi"), ringSubJet.deltaPhi, ringSubJet.ringPerp);
            }
          }

          // Fill ring histograms: (1D, lambda 2D correlations and jet 2D correlations):
          if (leadP.isValid) {
            if (familySwitches.doFamilyRing) {
              RING_OBSERVABLE_LEADP_FILL_LIST(APPLY_HISTO_FILL, "Ring") // Notice the usage of macros! If you change the variable names, this WILL break the code!
                                                                        // No, there should NOT be any ";" here! Read the macro definition for an explanation

              // Filling checks that rely on Eta>0 or Eta<0 checks for V0 and LeadingP eta:
              RING_OBSERVABLE_LEADP_ETA_SPLIT_FILL_LIST("Ring", leadPEtaPos, lambdaEtaPos);
            }
            histos.fill(HIST("IntegratedCuts/LeadP/pRingCuts"), 0, ringLeadP.ringObservable); // First bin of comparison
            if (doResamplingPassQA)
              histos.fill(HIST("IntegratedCuts/LeadP/pRingVsResamplingPass"), idxResampling, ringLeadP.ringObservable);
            if (v0InMassPeak)
              histos.fill(HIST("IntegratedCuts/LeadP/p2dRingCutsV0MassPeak"), 0, 1, ringLeadP.ringObservable); // Fills the inPeak bin
            else if (v0InMassWindow)
              histos.fill(HIST("IntegratedCuts/LeadP/p2dRingCutsV0MassPeak"), 0, 0, ringLeadP.ringObservable); // Fills the inWindow bin, which has a stricter interval than !v0InMassPeak
            histos.fill(HIST("IntegratedCuts/LeadP/hCountCuts"), 0);

          }
          if (familySwitches.doFamilyRing) {
            POLARIZATION_PROFILE_FILL_LIST(APPLY_HISTO_FILL, "Ring")
          }

          // Binary search using the pre-fetched axes for delta method of error bar estimation:
          int binPt = 0; // Dummy declarations
          int binMass = 0;
          int binDTheta = 0;
          if (leadJet.isValid) {
            if (familySwitches.doFamilyRing) {
              RING_OBSERVABLE_FILL_LIST(APPLY_HISTO_FILL, "Ring")
            }
            histos.fill(HIST("IntegratedCuts/LeadJet/pRingCuts"), 0, ringLeadJet.ringObservable);
            if (doResamplingPassQA)
              histos.fill(HIST("IntegratedCuts/LeadJet/pRingVsResamplingPass"), idxResampling, ringLeadJet.ringObservable);
            if (v0InMassPeak)
              histos.fill(HIST("IntegratedCuts/LeadJet/p2dRingCutsV0MassPeak"), 0, 1, ringLeadJet.ringObservable); // Fills the inPeak bin
            else if (v0InMassWindow)
              histos.fill(HIST("IntegratedCuts/LeadJet/p2dRingCutsV0MassPeak"), 0, 0, ringLeadJet.ringObservable); // Fills the inWindow bin, which has a stricter interval than !v0InMassPeak
            histos.fill(HIST("IntegratedCuts/LeadJet/hCountCuts"), 0);
            histos.fill(HIST("IntegratedCuts/LeadJet/pRingVsNV0s"), nLambdaLikeV0s, ringLeadJet.ringObservable);
            histos.fill(HIST("h2dNV0sVsCentrality"), nLambdaLikeV0s, centrality);

            // Properly fetching values as they are needed:
            binPt = mAxisPt->FindBin(v0pt);
            binMass = mAxisMass->FindBin(v0LambdaLikeMass);
            binDTheta = mAxisDTheta->FindBin(ringLeadJet.deltaTheta);
            if (familySwitches.doFamilyRing) {
              trackRing.addV0(ringLeadJet.ringObservable, binPt, binMass, binDTheta);
            }

            if (qaSwitches.doFakePolDiagnosticsQA) {
              // Measuring the AEE differentially (azimuthal efficiency effect, which causes different V0 topologies to be enhanced/suppressed)
              // AEE and HEE correlation:
              histos.fill(HIST("HelicityEfficiencyQA/LeadJet/h2dFakePolCounts_CosThetaVsPhiStar"), cosFakePol, phiStar);
              histos.fill(HIST("HelicityEfficiencyQA/LeadJet/p2dFakePolSignal_CosThetaVsPhiStar"), cosFakePol, phiStar, ringLeadJet.ringObservable);

              // AEE and DCA between daughters correlation:
              histos.fill(HIST("HelicityEfficiencyQA/LeadJet/h2dFakePolCounts_PhiStarVsDCAdau"), phiStar, dcaDau);
              histos.fill(HIST("HelicityEfficiencyQA/LeadJet/p2dFakePolSignal_PhiStarVsDCAdau"), phiStar, dcaDau, ringLeadJet.ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/LeadJet/p3dFakePolSignal_PhiStarVsDCAdauVsProxyEta"), phiStar, dcaDau, leadJet.eta, ringLeadJet.ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/LeadJet/p3dFakePolSignal_PhiStarVsDCAdauVsEtaLambda"), phiStar, dcaDau, v0eta, ringLeadJet.ringObservable);

              // DCA dau to PV correlation:
              // For proton-like daughter:
              histos.fill(HIST("HelicityEfficiencyQA/LeadJet/h2dFakePolCounts_PhiStarVsDCAProLike"), phiStar, protonLikeDCADauToPV);
              histos.fill(HIST("HelicityEfficiencyQA/LeadJet/p2dFakePolSignal_PhiStarVsDCAProLike"), phiStar, protonLikeDCADauToPV, ringLeadJet.ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/LeadJet/p3dFakePolSignal_PhiStarVsDCAProLikeVsProxyEta"), phiStar, protonLikeDCADauToPV, leadJet.eta, ringLeadJet.ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/LeadJet/p3dFakePolSignal_PhiStarVsDCAProLikeVsEtaLambda"), phiStar, protonLikeDCADauToPV, v0eta, ringLeadJet.ringObservable);
              // For pion-like daughter:
              histos.fill(HIST("HelicityEfficiencyQA/LeadJet/h2dFakePolCounts_PhiStarVsDCAPiLike"), phiStar, pionLikeDCADauToPV);
              histos.fill(HIST("HelicityEfficiencyQA/LeadJet/p2dFakePolSignal_PhiStarVsDCAPiLike"), phiStar, pionLikeDCADauToPV, ringLeadJet.ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/LeadJet/p3dFakePolSignal_PhiStarVsDCAPiLikeVsProxyEta"), phiStar, pionLikeDCADauToPV, leadJet.eta, ringLeadJet.ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/LeadJet/p3dFakePolSignal_PhiStarVsDCAPiLikeVsEtaLambda"), phiStar, pionLikeDCADauToPV, v0eta, ringLeadJet.ringObservable);

              histos.fill(HIST("HelicityEfficiencyQA/LeadJet/ProxyZ/pRingVsProxyZ"), leadJetZ, ringLeadJet.ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/LeadJet/ProxyZ/pRingOverProxyZ_VsProxyEta"), leadJet.eta, ringObservableOverLeadJetZ);
              histos.fill(HIST("HelicityEfficiencyQA/LeadJet/ProxyZ/pRingOverProxyZ_VsCosThetaHEE"), cosFakePol, ringObservableOverLeadJetZ);
              histos.fill(HIST("HelicityEfficiencyQA/LeadJet/ProxyZ/pRingOverProxyZ_VsPhiStar"), phiStar, ringObservableOverLeadJetZ);
              histos.fill(HIST("HelicityEfficiencyQA/LeadJet/ProxyZ/p2dRingOverProxyZ_VsProxyEtaVsCosThetaHEE"), leadJet.eta, cosFakePol, ringObservableOverLeadJetZ);
              histos.fill(HIST("HelicityEfficiencyQA/LeadJet/ProxyZ/p2dRingOverProxyZ_VsProxyEtaVsPhiStar"), leadJet.eta, phiStar, ringObservableOverLeadJetZ);

              // HEE resolved in three mass slices, so a peak-only structure can be told apart from a flat artifact:
              const bool isAEEPhiPos = deltaPhiLambdaProtonStar >= 0;
              histos.fill(HIST("HelicityEfficiencyQA/CosThetaHEEByMass/h2dCountsCosThetaHEEVsFullMass"), cosFakePol, v0LambdaLikeMass);
              histos.fill(HIST("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingCosThetaHEEVsFullMass"), cosFakePol, v0LambdaLikeMass, ringLeadJet.ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingCosThetaHEEVsMass"), cosFakePol, v0LambdaLikeMass, ringLeadJet.ringObservable);
              if (isAEEPhiPos)
                histos.fill(HIST("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingCosThetaHEEVsMassPosPhiAEE"), cosFakePol, v0LambdaLikeMass, ringLeadJet.ringObservable);
              else
                histos.fill(HIST("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingCosThetaHEEVsMassNegPhiAEE"), cosFakePol, v0LambdaLikeMass, ringLeadJet.ringObservable);
            } // end doFakePolDiagnosticsQA (leading-jet AEE/HEE)
          }
          if (subJet.isValid) {
            if (familySwitches.doFamilyRing) {
              RING_OBSERVABLE_2NDJET_FILL_LIST(APPLY_HISTO_FILL, "Ring")
            }
            histos.fill(HIST("IntegratedCuts/SubJet/pRingCuts"), 0, ringSubJet.ringObservable);
            if (doResamplingPassQA)
              histos.fill(HIST("IntegratedCuts/SubJet/pRingVsResamplingPass"), idxResampling, ringSubJet.ringObservable);
            if (v0InMassPeak)
              histos.fill(HIST("IntegratedCuts/SubJet/p2dRingCutsV0MassPeak"), 0, 1, ringSubJet.ringObservable); // Fills the inPeak bin
            else if (v0InMassWindow)
              histos.fill(HIST("IntegratedCuts/SubJet/p2dRingCutsV0MassPeak"), 0, 0, ringSubJet.ringObservable); // Fills the inWindow bin, which has a stricter interval than !v0InMassPeak
            histos.fill(HIST("IntegratedCuts/SubJet/hCountCuts"), 0);
          }

          if (qaSwitches.doFakePolDiagnosticsQA) {
            // Filling eta dependence QAs of the result (both for V0 and jet proxy):
            // Defining shared binning which depend on the V0 only:
            const int etaLambdaBin = lambdaEtaPos ? 3 : 4;
            if (leadJet.isValid) {
              histos.fill(HIST("EtaStudy/LeadJet/pRingEtaCuts"), 0, ringLeadJet.ringObservable);
              histos.fill(HIST("EtaStudy/LeadJet/pRingEtaCuts"), etaLambdaBin, ringLeadJet.ringObservable);

              // Bin indices for this proxy:
              // Bin 0: all, 1/2: proxy #eta sign, 3/4: #Lambda #eta sign, 5-8: joint,
              // 9/10: |#eta_{proxy}| >= R, 11-14: strict joint.
              const int etaProxyBin = leadJetEtaPos ? 1 : 2;
              const int etaProxyLambdaBin = (leadJetEtaPos ? 5 : 7) + (lambdaEtaPos ? 0 : 1);
              const int etaProxyStrictBin = leadJetEtaPos ? 9 : 10;
              const int etaProxyStrictLambdaBin = (leadJetEtaPos ? 11 : 13) + (lambdaEtaPos ? 0 : 1);

              histos.fill(HIST("EtaStudy/LeadJet/pRingEtaCuts"), etaProxyBin, ringLeadJet.ringObservable);
              histos.fill(HIST("EtaStudy/LeadJet/pRingEtaCuts"), etaProxyLambdaBin, ringLeadJet.ringObservable);

              // HEE study (helicity efficiency effect):
              histos.fill(HIST("EtaStudy/LeadJet/h2dFakePolCounts"), cosFakePol, 0);
              histos.fill(HIST("EtaStudy/LeadJet/h2dFakePolCounts"), cosFakePol, etaLambdaBin);
              histos.fill(HIST("EtaStudy/LeadJet/h2dFakePolCounts"), cosFakePol, etaProxyBin);
              histos.fill(HIST("EtaStudy/LeadJet/h2dFakePolCounts"), cosFakePol, etaProxyLambdaBin);
              // Same for signal:
              histos.fill(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsCosTheta"), cosFakePol, 0, ringLeadJet.ringObservable);
              histos.fill(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsCosTheta"), cosFakePol, etaLambdaBin, ringLeadJet.ringObservable);
              histos.fill(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsCosTheta"), cosFakePol, etaProxyBin, ringLeadJet.ringObservable);
              histos.fill(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsCosTheta"), cosFakePol, etaProxyLambdaBin, ringLeadJet.ringObservable);
              // Counter and ring accumulators for AEE study:
              histos.fill(HIST("EtaStudy/LeadJet/h2dCountsVsPhiStar"), phiStar, 0);
              histos.fill(HIST("EtaStudy/LeadJet/h2dCountsVsPhiStar"), phiStar, etaLambdaBin);
              histos.fill(HIST("EtaStudy/LeadJet/h2dCountsVsPhiStar"), phiStar, etaProxyBin);
              histos.fill(HIST("EtaStudy/LeadJet/h2dCountsVsPhiStar"), phiStar, etaProxyLambdaBin);
              histos.fill(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiStar"), phiStar, 0, ringLeadJet.ringObservable);
              histos.fill(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiStar"), phiStar, etaLambdaBin, ringLeadJet.ringObservable);
              histos.fill(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiStar"), phiStar, etaProxyBin, ringLeadJet.ringObservable);
              histos.fill(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiStar"), phiStar, etaProxyLambdaBin, ringLeadJet.ringObservable);

              histos.fill(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiAEE"), deltaPhiLambdaProtonStar, 0, ringLeadJet.ringObservable);
              histos.fill(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiAEE"), deltaPhiLambdaProtonStar, etaLambdaBin, ringLeadJet.ringObservable);
              histos.fill(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiAEE"), deltaPhiLambdaProtonStar, etaProxyBin, ringLeadJet.ringObservable);
              histos.fill(HIST("EtaStudy/LeadJet/p2dFakePolSignalVsPhiAEE"), deltaPhiLambdaProtonStar, etaProxyLambdaBin, ringLeadJet.ringObservable);

              // Dependencies for signal extraction (species-agnostic, see the bookings):
              histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/LeadJet/pRingObservableVsPhiAEE"), deltaPhiLambdaProtonStar, ringLeadJet.ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/LeadJet/p2dRingObservableVsPhiAEEVsMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringLeadJet.ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/LeadJet/p2dRingObservableVsPhiAEEVsProxyEta"), deltaPhiLambdaProtonStar, leadJet.eta, ringLeadJet.ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/LeadJet/p3dRingObservableVsPhiAEEVsProxyEtaVsMass"), deltaPhiLambdaProtonStar, leadJet.eta, v0LambdaLikeMass, ringLeadJet.ringObservable);
              if (leadJetEtaPos) {
                histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/LeadJet/p2dRingObservableVsPhiAEEVsMassPosProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringLeadJet.ringObservable);
                histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/LeadJet/p2dRingObservableVsPhiAEEVsProxyEtaSign"), deltaPhiLambdaProtonStar, 0.5, ringLeadJet.ringObservable);
                // Background stabilization using etaLambda differentially:
                if (deltaPhiLambdaProtonStar >= 0) {
                  histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/BkgStabilizationAttempt/p2dRingObservableEtaLambdaVsMassEtaProxyPosPhiAEEPos"), v0eta, v0LambdaLikeMass, ringLeadJet.ringObservable);
                  if (ringLeadJet.deltaPhi >= 0)
                    histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/BkgStabilizationAttempt/p2dRingObservableEtaLambdaVsMassEtaProxyPosPhiAEEPosDeltaPhiLeadJetPos"), v0eta, v0LambdaLikeMass, ringLeadJet.ringObservable);
                  else
                    histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/BkgStabilizationAttempt/p2dRingObservableEtaLambdaVsMassEtaProxyPosPhiAEEPosDeltaPhiLeadJetNeg"), v0eta, v0LambdaLikeMass, ringLeadJet.ringObservable);
                }
                // Background stabilization attempt with etaLambda sign only, PhiAEE differentially:
                if (v0eta >= 0) {
                  histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/BkgStabilizationAttempt/p2dRingObservablePhiAEEVsMassEtaProxyPosPhiEtaLambdaPos"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringLeadJet.ringObservable);
                  if (ringLeadJet.deltaPhi >= 0)
                    histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/BkgStabilizationAttempt/p2dRingObservablePhiAEEVsMassEtaProxyPosPhiEtaLambdaPosDeltaPhiLeadJetPos"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringLeadJet.ringObservable);
                  else
                    histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/BkgStabilizationAttempt/p2dRingObservablePhiAEEVsMassEtaProxyPosPhiEtaLambdaPosDeltaPhiLeadJetNeg"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringLeadJet.ringObservable);
                }
              } else {
                histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/LeadJet/p2dRingObservableVsPhiAEEVsMassNegProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringLeadJet.ringObservable);
                histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/LeadJet/p2dRingObservableVsPhiAEEVsProxyEtaSign"), deltaPhiLambdaProtonStar, -0.5, ringLeadJet.ringObservable);
                // Background stabilization using etaLambda differentially:
                if (deltaPhiLambdaProtonStar >= 0)
                  histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/BkgStabilizationAttempt/p2dRingObservableEtaLambdaVsMassEtaProxyNegPhiAEEPos"), v0eta, v0LambdaLikeMass, ringLeadJet.ringObservable);
                // Background stabilization attempt with etaLambda sign only, PhiAEE differentially:
                if (v0eta >= 0)
                  histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/BkgStabilizationAttempt/p2dRingObservablePhiAEEVsMassEtaProxyNegPhiEtaLambdaPos"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringLeadJet.ringObservable);
              }
              histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/ThreeBinMass/LeadJet/p2dRingObservableVsPhiAEEVs3BinMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringLeadJet.ringObservable);

              // Extra correlations test:
              histos.fill(HIST("HelicityEfficiencyQA/LeadJet/h2dFakePolCountsVsDeltaTheta"), cosFakePol, ringLeadJet.deltaTheta);
              histos.fill(HIST("HelicityEfficiencyQA/LeadJet/h2dFakePolCountsCosThetaVsLambdaPt"), cosFakePol, v0pt);
              // Split by proxy #eta sign:
              if (leadJetEtaPos)
                histos.fill(HIST("HelicityEfficiencyQA/LeadJet/h2dFakePolCountsVsDeltaThetaPosProxyEta"), cosFakePol, ringLeadJet.deltaTheta);
              else
                histos.fill(HIST("HelicityEfficiencyQA/LeadJet/h2dFakePolCountsVsDeltaThetaNegProxyEta"), cosFakePol, ringLeadJet.deltaTheta);

              if (pTLambdaCheck) {
                histos.fill(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtCut"), cosFakePol, 0);
                histos.fill(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtCut"), cosFakePol, etaLambdaBin);
                histos.fill(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtCut"), cosFakePol, etaProxyBin);
                histos.fill(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtCut"), cosFakePol, etaProxyLambdaBin);
                if (rapidityLambdaCheck) { // Stricter check
                  histos.fill(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtYCuts"), cosFakePol, 0);
                  histos.fill(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtYCuts"), cosFakePol, etaLambdaBin);
                  histos.fill(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtYCuts"), cosFakePol, etaProxyBin);
                  histos.fill(HIST("EtaStudy/LeadJet/h2dFakePolCountsLambdaPtYCuts"), cosFakePol, etaProxyLambdaBin);
                }
              }
              if (leadJetEtaStrict) { // |eta_{Jet}| >= R
                histos.fill(HIST("EtaStudy/LeadJet/pRingEtaCuts"), etaProxyStrictBin, ringLeadJet.ringObservable);
                histos.fill(HIST("EtaStudy/LeadJet/pRingEtaCuts"), etaProxyStrictLambdaBin, ringLeadJet.ringObservable);
              }
            }
            if (subJet.isValid) {
              // Same bin scheme as the leading jet above:
              const int etaProxyBin = subJetEtaPos ? 1 : 2;
              const int etaProxyLambdaBin = (subJetEtaPos ? 5 : 7) + (lambdaEtaPos ? 0 : 1);
              const int etaProxyStrictBin = subJetEtaPos ? 9 : 10;
              const int etaProxyStrictLambdaBin = (subJetEtaPos ? 11 : 13) + (lambdaEtaPos ? 0 : 1);

              histos.fill(HIST("EtaStudy/SubJet/pRingEtaCuts"), 0, ringSubJet.ringObservable);
              histos.fill(HIST("EtaStudy/SubJet/pRingEtaCuts"), etaLambdaBin, ringSubJet.ringObservable);
              histos.fill(HIST("EtaStudy/SubJet/pRingEtaCuts"), etaProxyBin, ringSubJet.ringObservable);
              histos.fill(HIST("EtaStudy/SubJet/pRingEtaCuts"), etaProxyLambdaBin, ringSubJet.ringObservable);
              if (subJetEtaStrict) { // |eta_{SubJet}| >= R
                histos.fill(HIST("EtaStudy/SubJet/pRingEtaCuts"), etaProxyStrictBin, ringSubJet.ringObservable);
                histos.fill(HIST("EtaStudy/SubJet/pRingEtaCuts"), etaProxyStrictLambdaBin, ringSubJet.ringObservable);
              }

              // Dependencies for signal extraction (AEE, species-agnostic):
              histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/SubJet/pRingObservableVsPhiAEE"), deltaPhiLambdaProtonStar, ringSubJet.ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/SubJet/p2dRingObservableVsPhiAEEVsMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringSubJet.ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/SubJet/p2dRingObservableVsPhiAEEVsProxyEta"), deltaPhiLambdaProtonStar, subJet.eta, ringSubJet.ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/SubJet/p3dRingObservableVsPhiAEEVsProxyEtaVsMass"), deltaPhiLambdaProtonStar, subJet.eta, v0LambdaLikeMass, ringSubJet.ringObservable);
              if (subJetEtaPos) {
                histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/SubJet/p2dRingObservableVsPhiAEEVsMassPosProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringSubJet.ringObservable);
                histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/SubJet/p2dRingObservableVsPhiAEEVsProxyEtaSign"), deltaPhiLambdaProtonStar, 0.5, ringSubJet.ringObservable);
              } else {
                histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/SubJet/p2dRingObservableVsPhiAEEVsMassNegProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringSubJet.ringObservable);
                histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/SubJet/p2dRingObservableVsPhiAEEVsProxyEtaSign"), deltaPhiLambdaProtonStar, -0.5, ringSubJet.ringObservable);
              }
              histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/ThreeBinMass/SubJet/p2dRingObservableVsPhiAEEVs3BinMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringSubJet.ringObservable);
            }
            if (leadP.isValid) {
              // Same bin scheme, without the strict variant (this axis stops at 9 bins):
              const int etaProxyBin = leadPEtaPos ? 1 : 2;
              const int etaProxyLambdaBin = (leadPEtaPos ? 5 : 7) + (lambdaEtaPos ? 0 : 1);

              histos.fill(HIST("EtaStudy/LeadP/pRingEtaCuts"), 0, ringLeadP.ringObservable);
              histos.fill(HIST("EtaStudy/LeadP/pRingEtaCuts"), etaLambdaBin, ringLeadP.ringObservable);
              histos.fill(HIST("EtaStudy/LeadP/pRingEtaCuts"), etaProxyBin, ringLeadP.ringObservable);
              histos.fill(HIST("EtaStudy/LeadP/pRingEtaCuts"), etaProxyLambdaBin, ringLeadP.ringObservable);
              if (v0InMassPeak) {
                histos.fill(HIST("EtaStudy/LeadP/p2dRingEtaCuts_MassSignalVsBackground"), 0, 1, ringLeadP.ringObservable);
                histos.fill(HIST("EtaStudy/LeadP/p2dRingEtaCuts_MassSignalVsBackground"), etaLambdaBin, 1, ringLeadP.ringObservable);
                histos.fill(HIST("EtaStudy/LeadP/p2dRingEtaCuts_MassSignalVsBackground"), etaProxyBin, 1, ringLeadP.ringObservable);
                histos.fill(HIST("EtaStudy/LeadP/p2dRingEtaCuts_MassSignalVsBackground"), etaProxyLambdaBin, 1, ringLeadP.ringObservable);
              } else if (v0InMassWindow) {
                histos.fill(HIST("EtaStudy/LeadP/p2dRingEtaCuts_MassSignalVsBackground"), 0, 0, ringLeadP.ringObservable);
                histos.fill(HIST("EtaStudy/LeadP/p2dRingEtaCuts_MassSignalVsBackground"), etaLambdaBin, 0, ringLeadP.ringObservable);
                histos.fill(HIST("EtaStudy/LeadP/p2dRingEtaCuts_MassSignalVsBackground"), etaProxyBin, 0, ringLeadP.ringObservable);
                histos.fill(HIST("EtaStudy/LeadP/p2dRingEtaCuts_MassSignalVsBackground"), etaProxyLambdaBin, 0, ringLeadP.ringObservable);
              }
              histos.fill(HIST("EtaStudy/LeadP/h2dFakePolCounts"), cosFakePol, 0);
              histos.fill(HIST("EtaStudy/LeadP/h2dFakePolCounts"), cosFakePol, etaLambdaBin);
              histos.fill(HIST("EtaStudy/LeadP/h2dFakePolCounts"), cosFakePol, etaProxyBin);
              histos.fill(HIST("EtaStudy/LeadP/h2dFakePolCounts"), cosFakePol, etaProxyLambdaBin);

              histos.fill(HIST("HelicityEfficiencyQA/LeadP/h2dFakePolCountsCosThetaVsLambdaPt"), cosFakePol, v0pt); // Understanding the population of events that has a leading particle (even though this does not need one to be calculated!)
              histos.fill(HIST("HelicityEfficiencyQA/LeadP/h2dFakePolCountsVsDeltaTheta"), cosFakePol, ringLeadP.deltaTheta);
              if (leadPEtaPos)
                histos.fill(HIST("HelicityEfficiencyQA/LeadP/h2dFakePolCountsVsDeltaThetaPosProxyEta"), cosFakePol, ringLeadP.deltaTheta);
              else
                histos.fill(HIST("HelicityEfficiencyQA/LeadP/h2dFakePolCountsVsDeltaThetaNegProxyEta"), cosFakePol, ringLeadP.deltaTheta);

              // Dependencies for signal extraction (AEE, species-agnostic):
              histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/LeadP/pRingObservableVsPhiAEE"), deltaPhiLambdaProtonStar, ringLeadP.ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/LeadP/p2dRingObservableVsPhiAEEVsMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringLeadP.ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/LeadP/p2dRingObservableVsPhiAEEVsProxyEta"), deltaPhiLambdaProtonStar, leadP.eta, ringLeadP.ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/LeadP/p3dRingObservableVsPhiAEEVsProxyEtaVsMass"), deltaPhiLambdaProtonStar, leadP.eta, v0LambdaLikeMass, ringLeadP.ringObservable);
              if (leadPEtaPos) {
                histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/LeadP/p2dRingObservableVsPhiAEEVsMassPosProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringLeadP.ringObservable);
                histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/LeadP/p2dRingObservableVsPhiAEEVsProxyEtaSign"), deltaPhiLambdaProtonStar, 0.5, ringLeadP.ringObservable);
              } else {
                histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/LeadP/p2dRingObservableVsPhiAEEVsMassNegProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringLeadP.ringObservable);
                histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/LeadP/p2dRingObservableVsPhiAEEVsProxyEtaSign"), deltaPhiLambdaProtonStar, -0.5, ringLeadP.ringObservable);
              }
              histos.fill(HIST("HelicityEfficiencyQA/PhiAEE/ThreeBinMass/LeadP/p2dRingObservableVsPhiAEEVs3BinMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringLeadP.ringObservable);
            }
          } // end doFakePolDiagnosticsQA (eta-dependence block)

          // Extra kinematic criteria for Lambda candidates (removes polarization background):
          if (kinematicLambdaCheck) {
            if (leadP.isValid) {
              if (familySwitches.doFamilyRingKinematicCuts) {
                RING_OBSERVABLE_LEADP_FILL_LIST(APPLY_HISTO_FILL, "RingKinematicCuts")

                // Filling checks that rely on Eta>0 or Eta<0 checks for V0 and LeadingP eta:
                RING_OBSERVABLE_LEADP_ETA_SPLIT_FILL_LIST("RingKinematicCuts", leadPEtaPos, lambdaEtaPos);
              }
              histos.fill(HIST("IntegratedCuts/LeadP/pRingCuts"), 1, ringLeadP.ringObservable);
              if (v0InMassPeak)
                histos.fill(HIST("IntegratedCuts/LeadP/p2dRingCutsV0MassPeak"), 1, 1, ringLeadP.ringObservable); // Fills the inPeak bin
              else if (v0InMassWindow)
                histos.fill(HIST("IntegratedCuts/LeadP/p2dRingCutsV0MassPeak"), 1, 0, ringLeadP.ringObservable); // Fills the inWindow bin, which has a stricter interval than !v0InMassPeak
              histos.fill(HIST("IntegratedCuts/LeadP/hCountCuts"), 1);
            }
            if (familySwitches.doFamilyRingKinematicCuts) {
              POLARIZATION_PROFILE_FILL_LIST(APPLY_HISTO_FILL, "RingKinematicCuts")
            }
            if (leadJet.isValid) {
              if (familySwitches.doFamilyRingKinematicCuts) {
                RING_OBSERVABLE_FILL_LIST(APPLY_HISTO_FILL, "RingKinematicCuts")
              }
              histos.fill(HIST("IntegratedCuts/LeadJet/pRingCuts"), 1, ringLeadJet.ringObservable);
              if (v0InMassPeak)
                histos.fill(HIST("IntegratedCuts/LeadJet/p2dRingCutsV0MassPeak"), 1, 1, ringLeadJet.ringObservable); // Fills the inPeak bin
              else if (v0InMassWindow)
                histos.fill(HIST("IntegratedCuts/LeadJet/p2dRingCutsV0MassPeak"), 1, 0, ringLeadJet.ringObservable); // Fills the inWindow bin, which has a stricter interval than !v0InMassPeak
              histos.fill(HIST("IntegratedCuts/LeadJet/hCountCuts"), 1);
              if (familySwitches.doFamilyRingKinematicCuts) {
                trackRingKinCuts.addV0(ringLeadJet.ringObservable, binPt, binMass, binDTheta);
              }
            }
            if (subJet.isValid) {
              if (familySwitches.doFamilyRingKinematicCuts) {
                RING_OBSERVABLE_2NDJET_FILL_LIST(APPLY_HISTO_FILL, "RingKinematicCuts")
              }
              histos.fill(HIST("IntegratedCuts/SubJet/pRingCuts"), 1, ringSubJet.ringObservable);
              if (v0InMassPeak)
                histos.fill(HIST("IntegratedCuts/SubJet/p2dRingCutsV0MassPeak"), 1, 1, ringSubJet.ringObservable); // Fills the inPeak bin
              else if (v0InMassWindow)
                histos.fill(HIST("IntegratedCuts/SubJet/p2dRingCutsV0MassPeak"), 1, 0, ringSubJet.ringObservable); // Fills the inWindow bin, which has a stricter interval than !v0InMassPeak
              histos.fill(HIST("IntegratedCuts/SubJet/hCountCuts"), 1);
            }
          }

          // Extra selection criteria on jet candidates:
          // (redundant for jets with R=0.4, but for jets with R<0.4 the leading jet may be farther in eta)
          if (kinematicLeadJetCheck) { // Already includes leadJet.isValid in the bool! (no need to check again)
            if (familySwitches.doFamilyJetKinematicCuts) {
              RING_OBSERVABLE_FILL_LIST(APPLY_HISTO_FILL, "JetKinematicCuts")
            }
            histos.fill(HIST("IntegratedCuts/LeadJet/pRingCuts"), 2, ringLeadJet.ringObservable);
            if (v0InMassPeak)
              histos.fill(HIST("IntegratedCuts/LeadJet/p2dRingCutsV0MassPeak"), 2, 1, ringLeadJet.ringObservable); // Fills the inPeak bin
            else if (v0InMassWindow)
              histos.fill(HIST("IntegratedCuts/LeadJet/p2dRingCutsV0MassPeak"), 2, 0, ringLeadJet.ringObservable); // Fills the inWindow bin, which has a stricter interval than !v0InMassPeak
            histos.fill(HIST("IntegratedCuts/LeadJet/hCountCuts"), 2);
            if (familySwitches.doFamilyJetKinematicCuts) {
              POLARIZATION_PROFILE_FILL_LIST(APPLY_HISTO_FILL, "JetKinematicCuts")
              trackJetKinCuts.addV0(ringLeadJet.ringObservable, binPt, binMass, binDTheta);
            }
          }

          // Extra selection criteria on both Lambda and jet candidates:
          if (kinematicLambdaCheck && kinematicLeadJetCheck) {
            if (familySwitches.doFamilyJetAndLambdaKinematicCuts) {
              RING_OBSERVABLE_FILL_LIST(APPLY_HISTO_FILL, "JetAndLambdaKinematicCuts")
            }
            histos.fill(HIST("IntegratedCuts/LeadJet/pRingCuts"), 3, ringLeadJet.ringObservable);
            if (v0InMassPeak)
              histos.fill(HIST("IntegratedCuts/LeadJet/p2dRingCutsV0MassPeak"), 3, 1, ringLeadJet.ringObservable); // Fills the inPeak bin
            else if (v0InMassWindow)
              histos.fill(HIST("IntegratedCuts/LeadJet/p2dRingCutsV0MassPeak"), 3, 0, ringLeadJet.ringObservable); // Fills the inWindow bin, which has a stricter interval than !v0InMassPeak
            histos.fill(HIST("IntegratedCuts/LeadJet/hCountCuts"), 3);
            if (familySwitches.doFamilyJetAndLambdaKinematicCuts) {
              POLARIZATION_PROFILE_FILL_LIST(APPLY_HISTO_FILL, "JetAndLambdaKinematicCuts")
              trackJetLambdaKinCuts.addV0(ringLeadJet.ringObservable, binPt, binMass, binDTheta);
            }
          }

          // Same variations for the leading particle and for the subleading jet:
          // (kinematicLeadPCheck already encodes leadP.isValid, so no extra gate needed here)
          if (kinematicLeadPCheck) {
            if (familySwitches.doFamilyJetKinematicCuts) {
              RING_OBSERVABLE_LEADP_FILL_LIST(APPLY_HISTO_FILL, "JetKinematicCuts")

              // Filling checks that rely on Eta>0 or Eta<0 checks for V0 and LeadingP eta:
              RING_OBSERVABLE_LEADP_ETA_SPLIT_FILL_LIST("JetKinematicCuts", leadPEtaPos, lambdaEtaPos);
            }
            histos.fill(HIST("IntegratedCuts/LeadP/pRingCuts"), 2, ringLeadP.ringObservable);
            if (v0InMassPeak)
              histos.fill(HIST("IntegratedCuts/LeadP/p2dRingCutsV0MassPeak"), 2, 1, ringLeadP.ringObservable); // Fills the inPeak bin
            else if (v0InMassWindow)
              histos.fill(HIST("IntegratedCuts/LeadP/p2dRingCutsV0MassPeak"), 2, 0, ringLeadP.ringObservable); // Fills the inWindow bin, which has a stricter interval than !v0InMassPeak
            histos.fill(HIST("IntegratedCuts/LeadP/hCountCuts"), 2);
          }
          if (kinematicSubJetCheck) {
            if (familySwitches.doFamilyJetKinematicCuts) {
              RING_OBSERVABLE_2NDJET_FILL_LIST(APPLY_HISTO_FILL, "JetKinematicCuts")
            }
            histos.fill(HIST("IntegratedCuts/SubJet/pRingCuts"), 2, ringSubJet.ringObservable);
            if (v0InMassPeak)
              histos.fill(HIST("IntegratedCuts/SubJet/p2dRingCutsV0MassPeak"), 2, 1, ringSubJet.ringObservable); // Fills the inPeak bin
            else if (v0InMassWindow)
              histos.fill(HIST("IntegratedCuts/SubJet/p2dRingCutsV0MassPeak"), 2, 0, ringSubJet.ringObservable); // Fills the inWindow bin, which has a stricter interval than !v0InMassPeak
            histos.fill(HIST("IntegratedCuts/SubJet/hCountCuts"), 2);
          }
          if (kinematicLambdaCheck && kinematicLeadPCheck) {
            if (familySwitches.doFamilyJetAndLambdaKinematicCuts) {
              RING_OBSERVABLE_LEADP_FILL_LIST(APPLY_HISTO_FILL, "JetAndLambdaKinematicCuts")

              // Filling checks that rely on Eta>0 or Eta<0 checks for V0 and LeadingP eta:
              RING_OBSERVABLE_LEADP_ETA_SPLIT_FILL_LIST("JetAndLambdaKinematicCuts", leadPEtaPos, lambdaEtaPos);
            }
            histos.fill(HIST("IntegratedCuts/LeadP/pRingCuts"), 3, ringLeadP.ringObservable);
            if (v0InMassPeak)
              histos.fill(HIST("IntegratedCuts/LeadP/p2dRingCutsV0MassPeak"), 3, 1, ringLeadP.ringObservable); // Fills the inPeak bin
            else if (v0InMassWindow)
              histos.fill(HIST("IntegratedCuts/LeadP/p2dRingCutsV0MassPeak"), 3, 0, ringLeadP.ringObservable); // Fills the inWindow bin, which has a stricter interval than !v0InMassPeak
            histos.fill(HIST("IntegratedCuts/LeadP/hCountCuts"), 3);
          }
          if (kinematicLambdaCheck && kinematicSubJetCheck) {
            if (familySwitches.doFamilyJetAndLambdaKinematicCuts) {
              RING_OBSERVABLE_2NDJET_FILL_LIST(APPLY_HISTO_FILL, "JetAndLambdaKinematicCuts")
            }
            histos.fill(HIST("IntegratedCuts/SubJet/pRingCuts"), 3, ringSubJet.ringObservable);
            if (v0InMassPeak)
              histos.fill(HIST("IntegratedCuts/SubJet/p2dRingCutsV0MassPeak"), 3, 1, ringSubJet.ringObservable); // Fills the inPeak bin
            else if (v0InMassWindow)
              histos.fill(HIST("IntegratedCuts/SubJet/p2dRingCutsV0MassPeak"), 3, 0, ringSubJet.ringObservable); // Fills the inWindow bin, which has a stricter interval than !v0InMassPeak
            histos.fill(HIST("IntegratedCuts/SubJet/hCountCuts"), 3);
          }
        } // end v0s loop

        // Flush trackers to the actual O2 histograms (via macros, so that O2 compiles properly):
        if (familySwitches.doFamilyRing) {
          FLUSH_DELTA_TRACKER("Ring", trackRing, mAxisPt, mAxisMass, mAxisDTheta)
        }
        if (familySwitches.doFamilyRingKinematicCuts) {
          FLUSH_DELTA_TRACKER("RingKinematicCuts", trackRingKinCuts, mAxisPt, mAxisMass, mAxisDTheta)
        }
        if (familySwitches.doFamilyJetKinematicCuts) {
          FLUSH_DELTA_TRACKER("JetKinematicCuts", trackJetKinCuts, mAxisPt, mAxisMass, mAxisDTheta)
        }
        if (familySwitches.doFamilyJetAndLambdaKinematicCuts) {
          FLUSH_DELTA_TRACKER("JetAndLambdaKinematicCuts", trackJetLambdaKinCuts, mAxisPt, mAxisMass, mAxisDTheta)
        }
      } // end collisions
    } // end of resampling loop for forceRandJet and forceDatalikeJet
  }

  PROCESS_SWITCH(lambdajetpolarizationionsderived, processPolarizationData, "Process derived data in Run 3 Data", true);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<lambdajetpolarizationionsderived>(cfgc)};
}

// Avoid macro leakage!
#undef APPLY_HISTO_FILL
