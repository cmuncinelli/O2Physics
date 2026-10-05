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
  X(FOLDER "/LeadJet/QA/hDeltaPhi", deltaPhiJet)                                                                                   \
  X(FOLDER "/LeadJet/QA/hDeltaPhiVsDeltaEta", deltaPhiJet, deltaEtaJet)                                                            \
  X(FOLDER "/LeadJet/QA/hDeltaTheta", deltaThetaJet)                                                                               \
  X(FOLDER "/LeadJet/QA/hCosDeltaTheta", cosDeltaThetaJet)                                                                         \
  X(FOLDER "/LeadJet/QA/hIntegrated", 0.)                                                                                          \
  X(FOLDER "/LeadJet/QA/hPtJet", leadingJetPt)                                                                                     \
  /* Lambda pT variation -- Youpeng's proposal */                                                                          \
  X(FOLDER "/LeadJet/QA/hLambdaPt", v0pt)                                                                                          \
  /* Counters */                                                                                                           \
  X(FOLDER "/LeadJet/QA/h2dDeltaPhiVsLambdaPt", deltaPhiJet, v0pt)                                                                 \
  X(FOLDER "/LeadJet/QA/h2dDeltaThetaVsLambdaPt", deltaThetaJet, v0pt)                                                             \
  X(FOLDER "/LeadJet/QA/hDeltaPhiVsLeadJetPhi", deltaPhiJet, leadingJetPhi)                                                        \
  /* Additional plots for instant gratification - 1D Profiles */                                                           \
  X(FOLDER "/LeadJet/hRingObservableCounts", ringObservable)                                                                       \
  X(FOLDER "/LeadJet/pRingObservableDeltaPhi", deltaPhiJet, ringObservable)                                                        \
  X(FOLDER "/LeadJet/pRingObservablePhiJet", leadingJetPhi, ringObservable)                                                        \
  X(FOLDER "/LeadJet/pRingObservablePhiLambda", v0phi, ringObservable)                                                             \
  X(FOLDER "/LeadJet/pRingObservableDeltaTheta", deltaThetaJet, ringObservable)                                                    \
  X(FOLDER "/LeadJet/EtaDependence/pRingObservableEtaLambda", v0eta, ringObservable)                                               \
  X(FOLDER "/LeadJet/EtaDependence/pRingObservableEtaJet", leadingJetEta, ringObservable)                                          \
  X(FOLDER "/LeadJet/EtaDependence/pRingObservableEtaJetHighEtaRes", leadingJetEta, ringObservable)                                \
  X(FOLDER "/LeadJet/pRingObservableIntegrated", 0., ringObservable)                                                               \
  X(FOLDER "/LeadJet/pRingObservableLambdaPt", v0pt, ringObservable)                                                               \
  X(FOLDER "/LeadJet/pRingObservableLeadJetPVz", collisionPVz, ringObservable)                                                     \
  X(FOLDER "/LeadJet/ProxyPtDependence/pRingVsPtJet", leadingJetPt, ringObservable)                                                \
  X(FOLDER "/LeadJet/ProxyPtDependence/pRingVsPtJetVsEtaJet", leadingJetPt, leadingJetEta, ringObservable)                         \
  X(FOLDER "/LeadJet/ProxyPtDependence/pRingVsPtJetVsEtaV0", leadingJetPt, v0eta, ringObservable)                                  \
  X(FOLDER "/LeadJet/ProxyPtDependence/pRingVsPtJetVsCentrality", leadingJetPt, centrality, ringObservable)                        \
  /* 2D Profiles */                                                                                                        \
  X(FOLDER "/LeadJet/p2dRingObservableDeltaPhiVsLambdaPt", deltaPhiJet, v0pt, ringObservable)                                      \
  X(FOLDER "/LeadJet/p2dRingObservableDeltaThetaVsLambdaPt", deltaThetaJet, v0pt, ringObservable)                                  \
  X(FOLDER "/LeadJet/p2dRingObservableDeltaPhiVsLeadJetPt", deltaPhiJet, leadingJetPt, ringObservable)                             \
  X(FOLDER "/LeadJet/p2dRingObservableDeltaThetaVsLeadJetPt", deltaThetaJet, leadingJetPt, ringObservable)                         \
  /* 1D Mass */                                                                                                            \
  X(FOLDER "/LeadJet/QA/hMass", v0LambdaLikeMass)                                                                                  \
  X(FOLDER "/LeadJet/QA/hRingObservableNumMass", v0LambdaLikeMass, ringObservable)                                                 \
  X(FOLDER "/LeadJet/hMassSigExtract", v0LambdaLikeMass)                                                                           \
  /* Counters */                                                                                                           \
  X(FOLDER "/LeadJet/QA/h2dDeltaPhiVsMass", deltaPhiJet, v0LambdaLikeMass)                                                         \
  X(FOLDER "/LeadJet/QA/h2dDeltaThetaVsMass", deltaThetaJet, v0LambdaLikeMass)                                                     \
  X(FOLDER "/LeadJet/QA/h3dDeltaPhiVsMassVsLambdaPt", deltaPhiJet, v0LambdaLikeMass, v0pt)                                         \
  X(FOLDER "/LeadJet/QA/h3dDeltaThetaVsMassVsLambdaPt", deltaThetaJet, v0LambdaLikeMass, v0pt)                                     \
  X(FOLDER "/LeadJet/QA/h3dDeltaPhiVsMassVsLeadJetPt", deltaPhiJet, v0LambdaLikeMass, leadingJetPt)                                \
  X(FOLDER "/LeadJet/QA/h3dDeltaThetaVsMassVsLeadJetPt", deltaThetaJet, v0LambdaLikeMass, leadingJetPt)                            \
  X(FOLDER "/LeadJet/QA/h3dDeltaPhiVsMassVsCent", deltaPhiJet, v0LambdaLikeMass, centrality)                                       \
  X(FOLDER "/LeadJet/QA/h3dDeltaThetaVsMassVsCent", deltaThetaJet, v0LambdaLikeMass, centrality)                                   \
  /* TProfile of Ring vs Mass */                                                                                           \
  X(FOLDER "/LeadJet/pRingObservableMass", v0LambdaLikeMass, ringObservable)                                                       \
  /* 2D Profiles: Angle vs Mass */                                                                                         \
  X(FOLDER "/LeadJet/p2dRingObservableDeltaPhiVsMass", deltaPhiJet, v0LambdaLikeMass, ringObservable)                              \
  X(FOLDER "/LeadJet/p2dRingObservableDeltaThetaVsMass", deltaThetaJet, v0LambdaLikeMass, ringObservable)                          \
  X(FOLDER "/LeadJet/p2dRingObservableEtaLambdaVsMass", v0eta, v0LambdaLikeMass, ringObservable)                                   \
  /* 2D Profiles: EtaProxy vs Mass */                                                                                      \
  X(FOLDER "/LeadJet/p2dRingObservableEtaLeadJetVsMass", leadingJetEta, v0LambdaLikeMass, ringObservable)                          \
  X(FOLDER "/LeadJet/h2dCounterEtaLeadJetVsMass", leadingJetEta, v0LambdaLikeMass)                                                 \
  /* 2D Profile: Ring vs Eta variables */                                                                                  \
  X(FOLDER "/LeadJet/EtaDependence/hCounterEtaLambdaMinusEtaJet", v0eta - leadingJetEta)                                           \
  X(FOLDER "/LeadJet/EtaDependence/pRingObservableEtaLambdaMinusEtaJet", v0eta - leadingJetEta, ringObservable)                    \
  X(FOLDER "/LeadJet/EtaDependence/p2dRingObservableEtaLambdaVsEtaJet", v0eta, leadingJetEta, ringObservable)                      \
  X(FOLDER "/LeadJet/EtaDependence/h2dCounterEtaLambdaVsEtaJet", v0eta, leadingJetEta)                                             \
  X(FOLDER "/LeadJet/EtaDependence/p2dRingObservableEtaLambdaVsEtaJet_FineBins", v0eta, leadingJetEta, ringObservable)             \
  X(FOLDER "/LeadJet/EtaDependence/h2dCounterEtaLambdaVsEtaJet_FineBins", v0eta, leadingJetEta)                                    \
  /* 3D Profiles: Angle vs Mass vs Lambda pT */                                                                            \
  X(FOLDER "/LeadJet/p3dRingObservableDeltaPhiVsMassVsLambdaPt", deltaPhiJet, v0LambdaLikeMass, v0pt, ringObservable)              \
  X(FOLDER "/LeadJet/p3dRingObservableDeltaThetaVsMassVsLambdaPt", deltaThetaJet, v0LambdaLikeMass, v0pt, ringObservable)          \
  /* 3D Profiles: Angle vs Mass vs Lead Jet pT */                                                                          \
  X(FOLDER "/LeadJet/p3dRingObservableDeltaPhiVsMassVsLeadJetPt", deltaPhiJet, v0LambdaLikeMass, leadingJetPt, ringObservable)     \
  X(FOLDER "/LeadJet/p3dRingObservableDeltaThetaVsMassVsLeadJetPt", deltaThetaJet, v0LambdaLikeMass, leadingJetPt, ringObservable) \
  /* 2D Profile: Mass vs Centrality */                                                                                     \
  X(FOLDER "/LeadJet/p2dRingObservableMassVsCent", v0LambdaLikeMass, centrality, ringObservable)                                   \
  /* 3D Profiles: Angle vs Mass vs Centrality */                                                                           \
  X(FOLDER "/LeadJet/p3dRingObservableDeltaPhiVsMassVsCent", deltaPhiJet, v0LambdaLikeMass, centrality, ringObservable)            \
  X(FOLDER "/LeadJet/p3dRingObservableDeltaThetaVsMassVsCent", deltaThetaJet, v0LambdaLikeMass, centrality, ringObservable)        \
  X(FOLDER "/LeadJet/pRingVsCentrality", centrality, ringObservable)                                                               \
  /* 2D Profiles of the ring: lab momentum planes */                                                                      \
  X(FOLDER "/RingMaps/p2dRingObservableVsPxPy", v0px, v0py, ringObservable)                                                        \
  X(FOLDER "/RingMaps/p2dRingObservableVsPzPx", v0pz, v0px, ringObservable)                                                        \
  X(FOLDER "/RingMaps/p2dRingObservableVsPyPz", v0py, v0pz, ringObservable)                                                        \
  /* Ring scalar on the AEE planes. R is invariant under the rotation, so only the binning should change */               \
  X(FOLDER "/RingMaps/p2dRingObservableVsPxAeePyAee", v0pxAee, v0pyAee, ringObservable)                                            \
  X(FOLDER "/RingMaps/p2dRingObservableVsPzPxAee", v0pz, v0pxAee, ringObservable)                                                  \
  X(FOLDER "/RingMaps/p2dRingObservableVsPyAeePz", v0pyAee, v0pz, ringObservable)                                                  \
  /* Ring scalar on the coordinates transverse to the jet (PrimeJet). */                                                  \
  X(FOLDER "/RingMaps/p2dRingObservableVsPxPyPrimeJet", v0pxPrimeJet, v0pyPrimeJet, ringObservable)                                \
  X(FOLDER "/RingMaps/p2dRingObservableVsPzPxPrimeJet", v0pzPrimeJet, v0pxPrimeJet, ringObservable)                                \
  X(FOLDER "/RingMaps/p2dRingObservableVsPyPzPrimeJet", v0pyPrimeJet, v0pzPrimeJet, ringObservable)                                \
  /* Ring projection kernel */                                                                                            \
  X(FOLDER "/LeadJet/RingKernel/p2dRingObservableCosDeltaThetaVsJetZ", cosDeltaThetaJet, jetZ, ringObservable)                    \
  X(FOLDER "/LeadJet/RingKernel/h2dCountsCosDeltaThetaVsJetZ", cosDeltaThetaJet, jetZ)                                            \
  X(FOLDER "/LeadJet/RingKernel/p3dRingObservableCosDeltaThetaVsJetZVsLambdaZ", cosDeltaThetaJet, jetZ, lambdaZ, ringObservable)  \
  /* Jet-transverse coordinate system's polarization maps. Filled here to use existing hasValidLeadingJet check */        \
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
  /* QAs on DeltaPhiJet -- Filed here as deltaPhiJet is only defined under hasValidLeadingJet */                          \
  X(FOLDER "/LeadJet/QA/pPxStarDeltaPhi", deltaPhiJet, polStarX)                                                          \
  X(FOLDER "/LeadJet/QA/pPyStarDeltaPhi", deltaPhiJet, polStarY)                                                          \
  X(FOLDER "/LeadJet/QA/pPzStarDeltaPhi", deltaPhiJet, polStarZ)                                                          \
  X(FOLDER "/LeadJet/QA/p2dPxStarDeltaPhiVsLambdaPt", deltaPhiJet, v0pt, polStarX)                                        \
  X(FOLDER "/LeadJet/QA/p2dPyStarDeltaPhiVsLambdaPt", deltaPhiJet, v0pt, polStarY)                                        \
  X(FOLDER "/LeadJet/QA/p2dPzStarDeltaPhiVsLambdaPt", deltaPhiJet, v0pt, polStarZ)                                        \
  /* KappaEff moments: kappa = 3 <num>/<den>, and products for the covariances */                                         \
  X(FOLDER "/LeadJet/KappaEff/pKappaNumLeadJetVsMass", v0LambdaLikeMass, kappaNumJet)                                     \
  X(FOLDER "/LeadJet/KappaEff/pKappaDenLeadJetVsMass", v0LambdaLikeMass, kappaDenJet)                                     \
  X(FOLDER "/LeadJet/KappaEff/pKappaNumTimesDenLeadJetVsMass", v0LambdaLikeMass, kappaNumJet * kappaDenJet)               \
  X(FOLDER "/LeadJet/KappaEff/pRingTimesDenLeadJetVsMass", v0LambdaLikeMass, ringObservable * kappaDenJet)                \
  /* CheapSigExtract mass region equivalents of KappaEff (bin 1 sideband, bin 2 peak, otherwise underflow) */             \
  X(FOLDER "/LeadJet/KappaEff/pRingLeadJetVsMassRegion", massRegion, ringObservable)                                      \
  X(FOLDER "/LeadJet/KappaEff/pKappaNumLeadJetVsMassRegion", massRegion, kappaNumJet)                                     \
  X(FOLDER "/LeadJet/KappaEff/pKappaDenLeadJetVsMassRegion", massRegion, kappaDenJet)                                     \
  X(FOLDER "/LeadJet/KappaEff/pKappaNumTimesDenLeadJetVsMassRegion", massRegion, kappaNumJet * kappaDenJet)
// (TODO: add counters for regular TH2Ds about centrality)

// For leading particle
#define RING_OBSERVABLE_LEADP_FILL_LIST(X, FOLDER)                                                            \
  X(FOLDER "/LeadP/QA/hDeltaPhiLeadP", deltaPhiLeadP)                                                         \
  X(FOLDER "/LeadP/QA/hDeltaThetaLeadP", deltaThetaLeadP)                                                     \
  X(FOLDER "/LeadP/QA/hPtLeadP", leadPPt)                                                                     \
  X(FOLDER "/LeadP/QA/hCosDeltaThetaLeadP", cosDeltaThetaLeadP)                                               \
  /* TProfile of Ring vs Mass */                                                                              \
  X(FOLDER "/LeadP/pRingObservableLeadPMass", v0LambdaLikeMass, ringObservableLeadP)                          \
  X(FOLDER "/LeadP/hRingObservableLeadPCounts", ringObservableLeadP)                                          \
  X(FOLDER "/LeadP/pRingObservableLeadPDeltaPhi", deltaPhiLeadP, ringObservableLeadP)                         \
  X(FOLDER "/LeadP/pRingObservableLeadPDeltaTheta", deltaThetaLeadP, ringObservableLeadP)                     \
  X(FOLDER "/LeadP/EtaDependence/pRingObservableEtaLambdaLeadP", v0eta, ringObservableLeadP)                  \
  X(FOLDER "/LeadP/EtaDependence/pRingObservableEtaLeadP", leadPEta, ringObservableLeadP)                     \
  X(FOLDER "/LeadP/EtaDependence/pRingObservableEtaLeadPHighEtaRes", leadPEta, ringObservableLeadP)           \
  X(FOLDER "/LeadP/pRingObservableLeadPIntegrated", 0., ringObservableLeadP)                                  \
  X(FOLDER "/LeadP/pRingObservableLeadPLambdaPt", v0pt, ringObservableLeadP)                                  \
  X(FOLDER "/LeadP/pRingObservableLeadPPVz", collisionPVz, ringObservableLeadP)                               \
  X(FOLDER "/LeadP/ProxyPtDependence/pRingVsPtLeadP", leadPPt, ringObservableLeadP)                           \
  X(FOLDER "/LeadP/ProxyPtDependence/pRingVsPtLeadPVsEtaLeadP", leadPPt, leadPEta, ringObservableLeadP)       \
  X(FOLDER "/LeadP/ProxyPtDependence/pRingVsPtLeadPVsEtaV0", leadPPt, v0eta, ringObservableLeadP)             \
  X(FOLDER "/LeadP/ProxyPtDependence/pRingVsPtLeadPVsCentrality", leadPPt, centrality, ringObservableLeadP)   \
  X(FOLDER "/LeadP/EtaDependence/p2dRingObservableEtaLambdaVsEtaLeadP", v0eta, leadPEta, ringObservableLeadP) \
  X(FOLDER "/LeadP/EtaDependence/h2dCounterEtaLambdaVsEtaLeadP", v0eta, leadPEta)                             \
  X(FOLDER "/RingMaps/p2dRingObservableLeadPVsPxPy", v0px, v0py, ringObservableLeadP)                         \
  X(FOLDER "/RingMaps/p2dRingObservableLeadPVsPzPx", v0pz, v0px, ringObservableLeadP)                         \
  X(FOLDER "/RingMaps/p2dRingObservableLeadPVsPyPz", v0py, v0pz, ringObservableLeadP)                         \
  /* Ring projection kernel */                                                                                \
  X(FOLDER "/LeadP/RingKernel/p2dRingObservableLeadPCosDeltaThetaVsLeadPZ", cosDeltaThetaLeadP, leadPZ, ringObservableLeadP) \
  X(FOLDER "/LeadP/RingKernel/h2dCountsLeadPCosDeltaThetaVsLeadPZ", cosDeltaThetaLeadP, leadPZ)               \
  X(FOLDER "/LeadP/RingKernel/p3dRingObservableLeadPCosDeltaThetaVsLeadPZVsLambdaZ", cosDeltaThetaLeadP, leadPZ, lambdaZ, ringObservableLeadP) \
  /* 2D Profiles: EtaProxy vs Mass */                                                                         \
  X(FOLDER "/LeadP/p2dRingObservableLeadPEtaLeadPVsMass", leadPEta, v0LambdaLikeMass, ringObservableLeadP)    \
  X(FOLDER "/LeadP/h2dCounterLeadPEtaLeadPVsMass", leadPEta, v0LambdaLikeMass)                                \
  /* KappaEff moments */                                                                                      \
  X(FOLDER "/LeadP/KappaEff/pKappaNumLeadPVsMass", v0LambdaLikeMass, kappaNumLeadP)                           \
  X(FOLDER "/LeadP/KappaEff/pKappaDenLeadPVsMass", v0LambdaLikeMass, kappaDenLeadP)                           \
  X(FOLDER "/LeadP/KappaEff/pKappaNumTimesDenLeadPVsMass", v0LambdaLikeMass, kappaNumLeadP * kappaDenLeadP)   \
  X(FOLDER "/LeadP/KappaEff/pRingTimesDenLeadPVsMass", v0LambdaLikeMass, ringObservableLeadP * kappaDenLeadP) \
  /* KappaEff with cheap signal extraction procedure */                                                       \
  X(FOLDER "/LeadP/KappaEff/pRingLeadPVsMassRegion", massRegion, ringObservableLeadP)                         \
  X(FOLDER "/LeadP/KappaEff/pKappaNumLeadPVsMassRegion", massRegion, kappaNumLeadP)                           \
  X(FOLDER "/LeadP/KappaEff/pKappaDenLeadPVsMassRegion", massRegion, kappaDenLeadP)                           \
  X(FOLDER "/LeadP/KappaEff/pKappaNumTimesDenLeadPVsMassRegion", massRegion, kappaNumLeadP * kappaDenLeadP)

// A macro that encapsulates all eta checks for leading particle and V0s, along with the fills
// Parameters:
//   FOLDER       -- histogram folder string (compile-time literal)
//   LEADP_IS_POS -- bool: leadPEtaPos
//   V0_IS_POS    -- bool: lambdaEtaPos
#define RING_OBSERVABLE_LEADP_ETA_SPLIT_FILL_LIST(FOLDER, LEADP_IS_POS, V0_IS_POS)                                      \
  do {                                                                                                                  \
    if (LEADP_IS_POS) {                                                                                                 \
      /* leadP marginal: positive side */                                                                               \
      APPLY_HISTO_FILL(FOLDER "/LeadP/ProxyPtDependence/pRingVsPtLeadP_PosEtaLeadP", leadPPt, ringObservableLeadP)            \
      if (V0_IS_POS) {                                                                                                  \
        /* V0 marginal: positive side */                                                                                \
        APPLY_HISTO_FILL(FOLDER "/LeadP/ProxyPtDependence/pRingVsPtLeadP_PosEtaV0", leadPPt, ringObservableLeadP)             \
        /* Joint: (+leadP, +V0) */                                                                                      \
        APPLY_HISTO_FILL(FOLDER "/LeadP/ProxyPtDependence/pRingVsPtLeadP_PosEtaLeadP_PosEtaV0", leadPPt, ringObservableLeadP) \
      } else {                                                                                                          \
        /* V0 marginal: negative side */                                                                                \
        APPLY_HISTO_FILL(FOLDER "/LeadP/ProxyPtDependence/pRingVsPtLeadP_NegEtaV0", leadPPt, ringObservableLeadP)             \
        /* Joint: (+leadP, -V0) */                                                                                      \
        APPLY_HISTO_FILL(FOLDER "/LeadP/ProxyPtDependence/pRingVsPtLeadP_PosEtaLeadP_NegEtaV0", leadPPt, ringObservableLeadP) \
      }                                                                                                                 \
    } else {                                                                                                            \
      /* leadP marginal: negative side */                                                                               \
      APPLY_HISTO_FILL(FOLDER "/LeadP/ProxyPtDependence/pRingVsPtLeadP_NegEtaLeadP", leadPPt, ringObservableLeadP)            \
      if (V0_IS_POS) {                                                                                                  \
        /* V0 marginal: positive side */                                                                                \
        APPLY_HISTO_FILL(FOLDER "/LeadP/ProxyPtDependence/pRingVsPtLeadP_PosEtaV0", leadPPt, ringObservableLeadP)             \
        /* Joint: (-leadP, +V0) */                                                                                      \
        APPLY_HISTO_FILL(FOLDER "/LeadP/ProxyPtDependence/pRingVsPtLeadP_NegEtaLeadP_PosEtaV0", leadPPt, ringObservableLeadP) \
      } else {                                                                                                          \
        /* V0 marginal: negative side */                                                                                \
        APPLY_HISTO_FILL(FOLDER "/LeadP/ProxyPtDependence/pRingVsPtLeadP_NegEtaV0", leadPPt, ringObservableLeadP)             \
        /* Joint: (-leadP, -V0) */                                                                                      \
        APPLY_HISTO_FILL(FOLDER "/LeadP/ProxyPtDependence/pRingVsPtLeadP_NegEtaLeadP_NegEtaV0", leadPPt, ringObservableLeadP) \
      }                                                                                                                 \
    }                                                                                                                   \
  } while (0)

// For subleading jet:
#define RING_OBSERVABLE_2NDJET_FILL_LIST(X, FOLDER)                                                                         \
  X(FOLDER "/SubJet/QA/hDeltaPhi2ndJet", deltaPhi2ndJet)                                                                    \
  X(FOLDER "/SubJet/QA/hDeltaTheta2ndJet", deltaTheta2ndJet)                                                                \
  X(FOLDER "/SubJet/QA/hCosDeltaTheta2ndJet", cosDeltaTheta2ndJet)                                                          \
  X(FOLDER "/SubJet/QA/hPt2ndJet", subleadingJetPt)                                                                         \
  /* TProfile of Ring vs Mass */                                                                                            \
  X(FOLDER "/SubJet/pRingObservable2ndJetMass", v0LambdaLikeMass, ringObservable2ndJet)                                     \
  X(FOLDER "/SubJet/hRingObservable2ndJetCounter", ringObservable2ndJet)                                                    \
  X(FOLDER "/SubJet/pRingObservable2ndJetDeltaPhi", deltaPhi2ndJet, ringObservable2ndJet)                                   \
  X(FOLDER "/SubJet/pRingObservable2ndJetDeltaTheta", deltaTheta2ndJet, ringObservable2ndJet)                               \
  X(FOLDER "/SubJet/EtaDependence/pRingObservableEtaLambda2ndJet", v0eta, ringObservable2ndJet)                             \
  X(FOLDER "/SubJet/EtaDependence/pRingObservableEta2ndJet", subleadingJetEta, ringObservable2ndJet)                        \
  X(FOLDER "/SubJet/pRingObservable2ndJetIntegrated", 0., ringObservable2ndJet)                                             \
  X(FOLDER "/SubJet/pRingObservable2ndJetLambdaPt", v0pt, ringObservable2ndJet)                                             \
  X(FOLDER "/SubJet/pRingObservableSubLeadPVz", collisionPVz, ringObservable2ndJet)                                         \
  X(FOLDER "/SubJet/ProxyPtDependence/pRingVsPt2ndJet", subleadingJetPt, ringObservable2ndJet)                              \
  X(FOLDER "/SubJet/ProxyPtDependence/pRingVsPt2ndJetVsEta2ndJet", subleadingJetPt, subleadingJetEta, ringObservable2ndJet) \
  X(FOLDER "/SubJet/ProxyPtDependence/pRingVsPt2ndJetVsEtaV0", subleadingJetPt, v0eta, ringObservable2ndJet)                \
  X(FOLDER "/SubJet/ProxyPtDependence/pRingVsPt2ndJetVsCentrality", subleadingJetPt, centrality, ringObservable2ndJet)      \
  X(FOLDER "/SubJet/EtaDependence/p2dRingObservableEtaLambdaVsEta2ndJet", v0eta, subleadingJetEta, ringObservable2ndJet)    \
  X(FOLDER "/SubJet/EtaDependence/h2dCounterEtaLambdaVsEta2ndJet", v0eta, subleadingJetEta)                                 \
  /* 2D Profiles: EtaProxy vs Mass */                                                                                       \
  X(FOLDER "/SubJet/p2dRingObservable2ndJetEta2ndJetVsMass", subleadingJetEta, v0LambdaLikeMass, ringObservable2ndJet)      \
  X(FOLDER "/SubJet/h2dCounter2ndJetEta2ndJetVsMass", subleadingJetEta, v0LambdaLikeMass)                                   \
  /* KappaEff moments */                                                                                                    \
  X(FOLDER "/SubJet/KappaEff/pKappaNumSubJetVsMass", v0LambdaLikeMass, kappaNum2ndJet)                                      \
  X(FOLDER "/SubJet/KappaEff/pKappaDenSubJetVsMass", v0LambdaLikeMass, kappaDen2ndJet)                                      \
  X(FOLDER "/SubJet/KappaEff/pKappaNumTimesDenSubJetVsMass", v0LambdaLikeMass, kappaNum2ndJet * kappaDen2ndJet)             \
  X(FOLDER "/SubJet/KappaEff/pRingTimesDenSubJetVsMass", v0LambdaLikeMass, ringObservable2ndJet * kappaDen2ndJet)           \
  /* KappaEff with cheap signal extraction procedure */                                                                     \
  X(FOLDER "/SubJet/KappaEff/pRingSubJetVsMassRegion", massRegion, ringObservable2ndJet)                                    \
  X(FOLDER "/SubJet/KappaEff/pKappaNumSubJetVsMassRegion", massRegion, kappaNum2ndJet)                                      \
  X(FOLDER "/SubJet/KappaEff/pKappaDenSubJetVsMassRegion", massRegion, kappaDen2ndJet)                                      \
  X(FOLDER "/SubJet/KappaEff/pKappaNumTimesDenSubJetVsMassRegion", massRegion, kappaNum2ndJet * kappaDen2ndJet)

#define POLARIZATION_PROFILE_FILL_LIST(X, FOLDER)                          \
  /* 1D TProfiles vs v0phi */                                              \
  X(FOLDER "/QA/pPxStarPhi", v0phiToFillHists, polStarX)                   \
  X(FOLDER "/QA/pPyStarPhi", v0phiToFillHists, polStarY)                   \
  X(FOLDER "/QA/pPzStarPhi", v0phiToFillHists, polStarZ)                   \
  /* 1D TProfiles vs the AEE angle */                                               \
  X(FOLDER "/QA/pPxStarPhiLambdaPhiProtonStar", deltaPhiLambdaProtonStar, polStarX) \
  X(FOLDER "/QA/pPyStarPhiLambdaPhiProtonStar", deltaPhiLambdaProtonStar, polStarY) \
  X(FOLDER "/QA/pPzStarPhiLambdaPhiProtonStar", deltaPhiLambdaProtonStar, polStarZ) \
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
  // Configurable<bool> doPPAnalysis{"doPPAnalysis", false, "if in pp, set to true. Default is HI"};
  // Configurable<bool> doJetProxy5dQA{"doJetProxy5dQA", false, "generates expensive THnSparse histograms for joint distribution QA of the jet proxies and collisions"};

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
    Configurable<int> nProxyResamples{"nProxyResamples", 1, "The amount of resamplings of jet direction per event. Use ONLY for forceRandJet and forceDatalikeJet"};
  } fakePolSwitches;

  // Configurable<float> jetRForSmudging{"jetRForSmudging", 0.4, "QA quantity: the chosen R scale for the jet direction smudge"}; // Superseeded by jetR: kept the same scale in analysis and QA
  Configurable<float> jetR{"jetR", 0.4f, "Radius of the jet"}; // Provide manually, please.
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
    ConfigurableAxis axisLambdaPx{"axisLambdaPx", {40, -3.0f, 3.0f}, "#Lambda p_{x} (GeV/c)"};
    ConfigurableAxis axisLambdaPy{"axisLambdaPy", {40, -3.0f, 3.0f}, "#Lambda p_{y} (GeV/c)"};
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
       0, 2, 4, 6, 8, 10, // 2 GeV bins
       15, 20,            // 5 GeV bins
       30, 40,            // 10 GeV bins
       60, 80,            // 20 GeV bins
       120, 160, 200},    // 40 GeV bins
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

  // Allocating one tracker per family so the accumulators are allocated only once:
  EventDeltaTracker trackRing, trackRingKinCuts, trackJetKinCuts, trackJetLambdaKinCuts;

  // Axis pointers for Delta Method binning (fetched once in init, declared once here)
  TAxis* mAxisPt = nullptr;
  TAxis* mAxisMass = nullptr;
  TAxis* mAxisDTheta = nullptr;

  // V0 phi selection:
  std::vector<float> v0PhiLimitsVec;

  void init(InitContext const&)
  {
    // Configuration validation:
    // These combinations would not crash otherwise, so they are rejected at init() time.
    const int nDistortionsOn = static_cast<int>(fakePolSwitches.forcePerpToJet) + static_cast<int>(fakePolSwitches.forceJetDirectionSmudge) +
                               static_cast<int>(fakePolSwitches.forceRandJet) + static_cast<int>(fakePolSwitches.forcePreviousJet) +
                               static_cast<int>(fakePolSwitches.forceDatalikeJet) + static_cast<int>(fakePolSwitches.doMixedEventProxies);
    if (nDistortionsOn > 1) // applyProxyDistortion() is an if/else chain, so extra switches are silently ignored
      LOG(fatal) << "fakePolSwitches: " << nDistortionsOn << " proxy distortions enabled at once. They are mutually exclusive -- "
                 << "applyProxyDistortion() would apply only the first and silently drop the rest.";
    if (fakePolSwitches.nProxyResamples > 1 && (fakePolSwitches.forcePreviousJet || fakePolSwitches.doMixedEventProxies))
      LOG(fatal) << "fakePolSwitches: nProxyResamples > 1 is only meaningful for forceRandJet/forceDatalikeJet. "
                 << "Previous-jet/Mixed-Event proxies do not change between resamplings, so every extra pass would double-count the same proxy.";
    if (excludeOutOfPeakQA && excludeInPeakQA) // Complementary selections (disjoint)
      LOG(fatal) << "excludeOutOfPeakQA and excludeInPeakQA are complementary: enabling both rejects every V0.";
    if (!analyseLambda && !analyseAntiLambda)
      LOG(fatal) << "analyseLambda and analyseAntiLambda are both false: no V0 would ever be analysed.";
    if (!familySwitches.doFamilyRing) // TODO: think of a smarter way of handling the axis getters for the DeltaMethod
      LOG(fatal) << "doFamilyRing must be on: the Delta Method accumulators take their binning from the Ring/ histograms.";
    if (std::abs((SidebandOuterNSigma - SidebandInnerNSigma) - PeakWindowNSigma) >= 1e-9)
      LOG(fatal) << "Sideband and peak windows must have the same width for the histograms here.";

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
      histosRingFamily.add((folder + "/LeadJet/QA/hDeltaPhiVsDeltaEta").c_str(), "#Delta#varphi_{jet};#Delta#varphi_{jet}; #eta_{#Lambda}-#eta_{Jet};Counts", kTH2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDeltaEtaCoarse});
      histosRingFamily.add((folder + "/LeadJet/QA/hDeltaPhiVsLeadJetPhi").c_str(), "#Delta#varphi_{jet};#varphi_{Jet};#varphi_{Jet};Counts", kTH2D, {axisConfigurations.axisPhi, axisConfigurations.axisDeltaPhi});
      histosRingFamily.add((folder + "/LeadJet/QA/hDeltaTheta").c_str(), "#Delta#theta_{jet};#Delta#theta_{jet};Counts", kTH1D, {axisConfigurations.axisDeltaTheta});
      histosRingFamily.add((folder + "/LeadJet/QA/hCosDeltaTheta").c_str(), "cos(#Delta#theta_{jet});cos(#Delta#theta_{jet});Counts", kTH1D, {axisConfigurations.axisCosTheta}); // Should actually be flat due to the geometry
      histosRingFamily.add((folder + "/LeadJet/QA/hIntegrated").c_str(), "Integrated counts; ;Counts", kTH1D, {{1, -0.5, 0.5}});

      histosRingFamily.add((folder + "/LeadP/QA/hDeltaPhiLeadP").c_str(), "#Delta#varphi_{LeadP};#Delta#varphi_{LeadP};Counts", kTH1D, {axisConfigurations.axisDeltaPhi});
      histosRingFamily.add((folder + "/LeadP/QA/hDeltaThetaLeadP").c_str(), "#Delta#theta_{LeadP};#Delta#theta_{LeadP};Counts", kTH1D, {axisConfigurations.axisDeltaTheta});
      histosRingFamily.add((folder + "/LeadP/QA/hCosDeltaThetaLeadP").c_str(), "cos(#Delta#theta_{LeadP});cos(#Delta#theta_{LeadP});Counts", kTH1D, {axisConfigurations.axisCosTheta}); // Should actually be flat due to the geometry
      histosRingFamily.add((folder + "/SubJet/QA/hDeltaPhi2ndJet").c_str(), "#Delta#varphi_{SubJet};#Delta#varphi_{SubJet};Counts", kTH1D, {axisConfigurations.axisDeltaPhi});
      histosRingFamily.add((folder + "/SubJet/QA/hDeltaTheta2ndJet").c_str(), "#Delta#theta_{SubJet};#Delta#theta_{SubJet};Counts", kTH1D, {axisConfigurations.axisDeltaTheta});
      histosRingFamily.add((folder + "/SubJet/QA/hCosDeltaTheta2ndJet").c_str(), "cos(#Delta#theta_{SubJet});cos(#Delta#theta_{SubJet});Counts", kTH1D, {axisConfigurations.axisCosTheta}); // Should actually be flat due to the geometry

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
      histosRingFamily.add((folder + "/QA/pPxStarPhiLambdaPhiProtonStar").c_str(), "<P_{#Lambda}^{*}>_{x} vs AEE angle;#phi_{#Lambda}-#phi_{p}^{*};<P_{#Lambda}^{*}>_{x}", kTProfile, {axisConfigurations.axisDeltaPhi});
      histosRingFamily.add((folder + "/QA/pPyStarPhiLambdaPhiProtonStar").c_str(), "<P_{#Lambda}^{*}>_{y} vs AEE angle;#phi_{#Lambda}-#phi_{p}^{*};<P_{#Lambda}^{*}>_{y}", kTProfile, {axisConfigurations.axisDeltaPhi});
      histosRingFamily.add((folder + "/QA/pPzStarPhiLambdaPhiProtonStar").c_str(), "<P_{#Lambda}^{*}>_{z} vs AEE angle;#phi_{#Lambda}-#phi_{p}^{*};<P_{#Lambda}^{*}>_{z}", kTProfile, {axisConfigurations.axisDeltaPhi});
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
      histosRingFamily.add((folder + "/RingMaps/p2dRingObservableVsPxPy").c_str(), "<#it{R}> vs (p_{x}^{#Lambda},p_{y}^{#Lambda});p_{x}^{#Lambda} (GeV/c);p_{y}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/RingMaps/p2dRingObservableVsPzPx").c_str(), "<#it{R}> vs (p_{z}^{#Lambda},p_{x}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{x}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/RingMaps/p2dRingObservableVsPyPz").c_str(), "<#it{R}> vs (p_{y}^{#Lambda},p_{z}^{#Lambda});p_{y}^{#Lambda} (GeV/c);p_{z}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPz});
      // On the AEE planes:
      histosRingFamily.add((folder + "/RingMaps/p2dRingObservableVsPxAeePyAee").c_str(), "<#it{R}> vs (p_{x,AEE}^{#Lambda},p_{y,AEE}^{#Lambda});p_{x,AEE}^{#Lambda} (GeV/c);p_{y,AEE}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/RingMaps/p2dRingObservableVsPzPxAee").c_str(), "<#it{R}> vs (p_{z}^{#Lambda},p_{x,AEE}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{x,AEE}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/RingMaps/p2dRingObservableVsPyAeePz").c_str(), "<#it{R}> vs (p_{y,AEE}^{#Lambda},p_{z}^{#Lambda});p_{y,AEE}^{#Lambda} (GeV/c);p_{z}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPz});
      // On the jet-frame planes:
      histosRingFamily.add((folder + "/RingMaps/p2dRingObservableVsPxPyPrimeJet").c_str(), "<#it{R}> vs (p_{x'Jet}^{#Lambda},p_{y'Jet}^{#Lambda});p_{x'Jet}^{#Lambda} (GeV/c);p_{y'Jet}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/RingMaps/p2dRingObservableVsPzPxPrimeJet").c_str(), "<#it{R}> vs (p_{z'Jet}^{#Lambda},p_{x'Jet}^{#Lambda});p_{z'Jet}^{#Lambda} (GeV/c);p_{x'Jet}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/RingMaps/p2dRingObservableVsPyPzPrimeJet").c_str(), "<#it{R}> vs (p_{y'Jet}^{#Lambda},p_{z'Jet}^{#Lambda});p_{y'Jet}^{#Lambda} (GeV/c);p_{z'Jet}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      // For LeadP estimators:
      histosRingFamily.add((folder + "/RingMaps/p2dRingObservableLeadPVsPxPy").c_str(), "<#it{R}>_{LeadP} vs (p_{x}^{#Lambda},p_{y}^{#Lambda});p_{x}^{#Lambda} (GeV/c);p_{y}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/RingMaps/p2dRingObservableLeadPVsPzPx").c_str(), "<#it{R}>_{LeadP} vs (p_{z}^{#Lambda},p_{x}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{x}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPRot});
      histosRingFamily.add((folder + "/RingMaps/p2dRingObservableLeadPVsPyPz").c_str(), "<#it{R}>_{LeadP} vs (p_{y}^{#Lambda},p_{z}^{#Lambda});p_{y}^{#Lambda} (GeV/c);p_{z}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPz});

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
      histosRingFamily.add((folder + "/LeadJet/pRingObservablePhiJet").c_str(), "<#it{R}> vs #varphi_{jet};#varphi_{jet};<#it{R}>", kTProfile, {axisConfigurations.axisPhi});
      histosRingFamily.add((folder + "/LeadJet/pRingObservablePhiLambda").c_str(), "<#it{R}> vs #varphi_{#Lambda};#varphi_{#Lambda};<#it{R}>", kTProfile, {axisConfigurations.axisPhi});
      // ===============================
      // Ring projection kernel
      // The two variables <R> depends on analytically, plus the Lambda direction cosine in the 3D version,
      // which is what lets the eta_Lambda sample be symmetrised downstream instead of at fill time.
      // ===============================
      // Binned in the two variables the projection is analytically linear in -- cos(DeltaTheta) and the
      // proxy direction cosine t_z -- rather than in DeltaTheta and eta. Any fitting of the resulting
      // surface belongs downstream, in the post-processing macros.
      histosRingFamily.add((folder + "/LeadJet/RingKernel/p2dRingObservableCosDeltaThetaVsJetZ").c_str(), "<#it{R}> vs (cos#Delta#theta_{jet},#hat{t}_{z});cos#Delta#theta_{jet};#hat{t}_{z};<#it{R}>", kTProfile2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisJetZ});
      histosRingFamily.add((folder + "/LeadJet/RingKernel/h2dCountsCosDeltaThetaVsJetZ").c_str(), "Counts vs (cos#Delta#theta_{jet},#hat{t}_{z});cos#Delta#theta_{jet};#hat{t}_{z};Counts", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisJetZ});
      histosRingFamily.add((folder + "/LeadJet/RingKernel/p3dRingObservableCosDeltaThetaVsJetZVsLambdaZ").c_str(), "<#it{R}> vs (cos#Delta#theta_{jet},#hat{t}_{z},cos#theta_{#Lambda});cos#Delta#theta_{jet};#hat{t}_{z};cos#theta_{#Lambda}", kTProfile3D, {axisConfigurations.axisCosThetaCoarse, axisConfigurations.axisJetZ, axisConfigurations.axisLambdaZ});

      // LeadP proxy. B_phi knows nothing about the reference axis, so this must return the same kernel as the jet version -- with a wider t_z lever.
      histosRingFamily.add((folder + "/LeadP/RingKernel/p2dRingObservableLeadPCosDeltaThetaVsLeadPZ").c_str(), "<#it{R}>_{LeadP} vs (cos#Delta#theta_{LeadP},#hat{t}_{z}^{LeadP});cos#Delta#theta_{LeadP};#hat{t}_{z}^{LeadP};<#it{R}>", kTProfile2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisLeadPZ});
      histosRingFamily.add((folder + "/LeadP/RingKernel/h2dCountsLeadPCosDeltaThetaVsLeadPZ").c_str(), "Counts vs (cos#Delta#theta_{LeadP},#hat{t}_{z}^{LeadP});cos#Delta#theta_{LeadP};#hat{t}_{z}^{LeadP};Counts", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisLeadPZ});
      histosRingFamily.add((folder + "/LeadP/RingKernel/p3dRingObservableLeadPCosDeltaThetaVsLeadPZVsLambdaZ").c_str(), "<#it{R}>_{LeadP} vs (cos#Delta#theta_{LeadP},#hat{t}_{z}^{LeadP},cos#theta_{#Lambda});cos#Delta#theta_{LeadP};#hat{t}_{z}^{LeadP};cos#theta_{#Lambda}", kTProfile3D, {axisConfigurations.axisCosThetaCoarse, axisConfigurations.axisLeadPZ, axisConfigurations.axisLambdaZ});

      histosRingFamily.add((folder + "/LeadJet/pRingObservableDeltaTheta").c_str(), "<#it{R}> vs #Delta#theta_{jet};#Delta#theta_{jet};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaTheta});
      histosRingFamily.add((folder + "/LeadJet/pRingObservableIntegrated").c_str(), "Integrated <#it{R}>; ;<#it{R}>", kTProfile, {{1, -0.5, 0.5}});
      histosRingFamily.add((folder + "/LeadJet/pRingObservableLambdaPt").c_str(), "<#it{R}> vs #it{p}_{T}^{#Lambda};#it{p}_{T}^{#Lambda} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisPt});

      // Ring vs Jet proxy pT:
      histosRingFamily.add((folder + "/LeadJet/ProxyPtDependence/pRingVsPtJet").c_str(), "<#it{R}> vs Jet #it{p}_{T};#it{p}_{T}^{Jet} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
      histosRingFamily.add((folder + "/LeadP/ProxyPtDependence/pRingVsPtLeadP").c_str(), "<#it{R}> vs LeadP #it{p}_{T};#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
      histosRingFamily.add((folder + "/SubJet/ProxyPtDependence/pRingVsPt2ndJet").c_str(), "<#it{R}> vs SubJet #it{p}_{T};#it{p}_{T}^{SubJet} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
      // And some counters to be aware of the amount of Lambdas (and jets) in each pT interval:
      histosRingFamily.add((folder + "/LeadJet/QA/hPtJet").c_str(), "Jet #it{p}_{T};#it{p}_{T}^{Jet} (GeV/c);Counts", kTH1D, {axisConfigurations.axisJetPt});
      histosRingFamily.add((folder + "/LeadP/QA/hPtLeadP").c_str(), "LeadP #it{p}_{T};#it{p}_{T}^{LeadP} (GeV/c);Counts", kTH1D, {axisConfigurations.axisJetPt});
      histosRingFamily.add((folder + "/SubJet/QA/hPt2ndJet").c_str(), "SubJet #it{p}_{T};#it{p}_{T}^{SubJet} (GeV/c);Counts", kTH1D, {axisConfigurations.axisJetPt});

      // Splitting into positive and negative eta contributions:
      histosRingFamily.add((folder + "/LeadJet/ProxyPtDependence/pRingVsPtJetVsEtaJet").c_str(), "<#it{R}> vs Jet #it{p}_{T} vs #eta_{Jet};#it{p}_{T}^{Jet} (GeV/c);#eta_{Jet};<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, {2, -0.9, 0.9}});
      histosRingFamily.add((folder + "/LeadP/ProxyPtDependence/pRingVsPtLeadPVsEtaLeadP").c_str(), "<#it{R}> vs LeadP #it{p}_{T} vs #eta_{LeadP};#it{p}_{T}^{LeadP} (GeV/c);#eta_{LeadP};<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, {2, -0.9, 0.9}});
      histosRingFamily.add((folder + "/SubJet/ProxyPtDependence/pRingVsPt2ndJetVsEta2ndJet").c_str(), "<#it{R}> vs SubJet #it{p}_{T} vs #eta_{SubJet};#it{p}_{T}^{SubJet} (GeV/c);#eta_{SubJet};<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, {2, -0.9, 0.9}});
      // For each Lambda's eta:
      histosRingFamily.add((folder + "/LeadJet/ProxyPtDependence/pRingVsPtJetVsEtaV0").c_str(), "<#it{R}> vs Jet #it{p}_{T} vs #eta_{V0};#it{p}_{T}^{Jet} (GeV/c);#eta_{V0};<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, {2, -0.9, 0.9}});
      histosRingFamily.add((folder + "/LeadP/ProxyPtDependence/pRingVsPtLeadPVsEtaV0").c_str(), "<#it{R}> vs LeadP #it{p}_{T} vs #eta_{V0};#it{p}_{T}^{LeadP} (GeV/c);#eta_{V0};<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, {2, -0.9, 0.9}});
      histosRingFamily.add((folder + "/SubJet/ProxyPtDependence/pRingVsPt2ndJetVsEtaV0").c_str(), "<#it{R}> vs SubJet #it{p}_{T} vs #eta_{V0};#it{p}_{T}^{SubJet} (GeV/c);#eta_{V0};<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, {2, -0.9, 0.9}});

      // Rasterizing, only for LeadP the TProfile2D into two TProfile 1Ds (easier to draw with "same")
      histosRingFamily.add((folder + "/LeadP/ProxyPtDependence/pRingVsPtLeadP_PosEtaLeadP").c_str(), "<#it{R}> vs LeadP #it{p}_{T} (#eta_{LeadP}>0);#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
      histosRingFamily.add((folder + "/LeadP/ProxyPtDependence/pRingVsPtLeadP_NegEtaLeadP").c_str(), "<#it{R}> vs LeadP #it{p}_{T} (#eta_{LeadP}<0);#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
      // V0 eta:
      histosRingFamily.add((folder + "/LeadP/ProxyPtDependence/pRingVsPtLeadP_PosEtaV0").c_str(), "<#it{R}> vs LeadP #it{p}_{T} (#eta_{V0}>0);#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
      histosRingFamily.add((folder + "/LeadP/ProxyPtDependence/pRingVsPtLeadP_NegEtaV0").c_str(), "<#it{R}> vs LeadP #it{p}_{T} (#eta_{V0}<0);#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});

      // Looking at V0Eta and JetEta combinations (only for LeadP):
      histosRingFamily.add((folder + "/LeadP/ProxyPtDependence/pRingVsPtLeadP_PosEtaLeadP_PosEtaV0").c_str(), "<#it{R}> vs LeadP #it{p}_{T} (#eta_{LeadP}>0, #eta_{V0}>0);#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
      histosRingFamily.add((folder + "/LeadP/ProxyPtDependence/pRingVsPtLeadP_NegEtaLeadP_PosEtaV0").c_str(), "<#it{R}> vs LeadP #it{p}_{T} (#eta_{LeadP}<0, #eta_{V0}>0);#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
      histosRingFamily.add((folder + "/LeadP/ProxyPtDependence/pRingVsPtLeadP_PosEtaLeadP_NegEtaV0").c_str(), "<#it{R}> vs LeadP #it{p}_{T} (#eta_{LeadP}>0, #eta_{V0}<0);#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
      histosRingFamily.add((folder + "/LeadP/ProxyPtDependence/pRingVsPtLeadP_NegEtaLeadP_NegEtaV0").c_str(), "<#it{R}> vs LeadP #it{p}_{T} (#eta_{LeadP}<0, #eta_{V0}<0);#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});

      // Integrated <R> Vs ProxyPt Vs Centrality:
      histosRingFamily.add((folder + "/LeadJet/ProxyPtDependence/pRingVsPtJetVsCentrality").c_str(), "<#it{R}> vs Jet #it{p}_{T} vs Centrality;#it{p}_{T}^{Jet} (GeV/c);Centrality(%);<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, axisConfigurations.axisCentrality});
      histosRingFamily.add((folder + "/LeadP/ProxyPtDependence/pRingVsPtLeadPVsCentrality").c_str(), "<#it{R}> vs LeadP #it{p}_{T} vs Centrality;#it{p}_{T}^{LeadP} (GeV/c);Centrality(%);<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, axisConfigurations.axisCentrality});
      histosRingFamily.add((folder + "/SubJet/ProxyPtDependence/pRingVsPt2ndJetVsCentrality").c_str(), "<#it{R}> vs SubJet #it{p}_{T} vs Centrality;#it{p}_{T}^{SubJet} (GeV/c);Centrality(%);<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, axisConfigurations.axisCentrality});

      // Understanding eta dependence seen in pRingEtaCuts:
      histosRingFamily.add((folder + "/LeadJet/EtaDependence/pRingObservableEtaLambda").c_str(), "<#it{R}> vs #eta_{#Lambda};#eta_{#Lambda};<#it{R}>", kTProfile, {axisConfigurations.axisV0EtaCoarse});
      histosRingFamily.add((folder + "/LeadJet/EtaDependence/pRingObservableEtaJet").c_str(), "<#it{R}> vs #eta_{Jet};#eta_{Jet};<#it{R}>", kTProfile, {axisConfigurations.axisEtaCoarse});
      histosRingFamily.add((folder + "/LeadJet/EtaDependence/pRingObservableEtaJetHighEtaRes").c_str(), "<#it{R}> vs #eta_{Jet};#eta_{Jet};<#it{R}>", kTProfile, {axisConfigurations.axisEta});

      histosRingFamily.add((folder + "/SubJet/EtaDependence/pRingObservableEtaLambda2ndJet").c_str(), "<#it{R}> vs #eta_{#Lambda} (SubJet);#eta_{#Lambda};<#it{R}>", kTProfile, {axisConfigurations.axisV0EtaCoarse});
      histosRingFamily.add((folder + "/SubJet/EtaDependence/pRingObservableEta2ndJet").c_str(), "<#it{R}> vs #eta_{SubJet};#eta_{SubJet};<#it{R}>", kTProfile, {axisConfigurations.axisEtaCoarse});

      histosRingFamily.add((folder + "/LeadP/EtaDependence/pRingObservableEtaLambdaLeadP").c_str(), "<#it{R}> vs #eta_{#Lambda} (LeadP);#eta_{#Lambda};<#it{R}>", kTProfile, {axisConfigurations.axisV0EtaCoarse});
      histosRingFamily.add((folder + "/LeadP/EtaDependence/pRingObservableEtaLeadP").c_str(), "<#it{R}> vs #eta_{LeadP};#eta_{LeadP};<#it{R}>", kTProfile, {axisConfigurations.axisEtaCoarse});
      histosRingFamily.add((folder + "/LeadP/EtaDependence/pRingObservableEtaLeadPHighEtaRes").c_str(), "<#it{R}> vs #eta_{LeadP};#eta_{LeadP};<#it{R}>", kTProfile, {axisConfigurations.axisEta});
      // For the leading particle:
      histosRingFamily.add((folder + "/LeadP/hRingObservableLeadPCounts").c_str(), "Counts vs <#it{R}>_{LeadP};<#it{R}>; Counts", kTH1D, {axisConfigurations.axisRingCounts});
      histosRingFamily.add((folder + "/LeadP/pRingObservableLeadPDeltaPhi").c_str(), "<#it{R}> vs #Delta#varphi_{LeadP};#Delta#varphi_{LeadP};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      histosRingFamily.add((folder + "/LeadP/pRingObservableLeadPDeltaTheta").c_str(), "<#it{R}> vs #Delta#theta_{LeadP};#Delta#theta_{LeadP};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaTheta});
      histosRingFamily.add((folder + "/LeadP/pRingObservableLeadPIntegrated").c_str(), "Integrated <#it{R}> (LeadP); ;<#it{R}>", kTProfile, {{1, -0.5, 0.5}});
      histosRingFamily.add((folder + "/LeadP/pRingObservableLeadPLambdaPt").c_str(), "<#it{R}> vs #it{p}_{T}^{#Lambda} (LeadP);#it{p}_{T}^{#Lambda} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisPt});
      // For the second-to-leading jet:
      histosRingFamily.add((folder + "/SubJet/hRingObservable2ndJetCounter").c_str(), "Counts vs <#it{R}>_{SubJet};<#it{R}>; Counts", kTH1D, {axisConfigurations.axisRingCounts});
      histosRingFamily.add((folder + "/SubJet/pRingObservable2ndJetDeltaPhi").c_str(), "<#it{R}> vs #Delta#varphi_{SubJet};#Delta#varphi_{SubJet};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      histosRingFamily.add((folder + "/SubJet/pRingObservable2ndJetDeltaTheta").c_str(), "<#it{R}> vs #Delta#theta_{SubJet};#Delta#theta_{SubJet};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaTheta});
      histosRingFamily.add((folder + "/SubJet/pRingObservable2ndJetIntegrated").c_str(), "Integrated <#it{R}> (SubJet); ;<#it{R}>", kTProfile, {{1, -0.5, 0.5}});
      histosRingFamily.add((folder + "/SubJet/pRingObservable2ndJetLambdaPt").c_str(), "<#it{R}> vs #it{p}_{T}^{#Lambda} (SubJet);#it{p}_{T}^{#Lambda} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisPt});

      // For the Zvtx dependence:
      histosRingFamily.add((folder + "/LeadJet/pRingObservableLeadJetPVz").c_str(), "<#it{R}>_{LeadJet} vs PVz;PVz (cm);<#it{R}>", kTProfile, {axisConfigurations.axisPVzCoarse});
      histosRingFamily.add((folder + "/SubJet/pRingObservableSubLeadPVz").c_str(), "<#it{R}>_{SubLead} vs PVz;PVz (cm);<#it{R}>", kTProfile, {axisConfigurations.axisPVzCoarse});
      histosRingFamily.add((folder + "/LeadP/pRingObservableLeadPPVz").c_str(), "<#it{R}>_{LeadP} vs PVz;PVz (cm);<#it{R}>", kTProfile, {axisConfigurations.axisPVzCoarse});
      // ===============================
      // 2D TProfiles (Lambda correlations)
      // ===============================
      histosRingFamily.add((folder + "/LeadJet/p2dRingObservableDeltaPhiVsLambdaPt").c_str(), "<#it{R}> vs #Delta#varphi_{jet} vs #it{p}_{T}^{#Lambda};#Delta#varphi_{jet};#it{p}_{T}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisPt});
      histosRingFamily.add((folder + "/LeadJet/p2dRingObservableDeltaThetaVsLambdaPt").c_str(), "<#it{R}> vs #Delta#theta_{jet} vs #it{p}_{T}^{#Lambda};#Delta#theta_{jet};#it{p}_{T}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisPt});
      // ===============================
      // 2D TProfiles (Jet correlations)
      // ===============================
      histosRingFamily.add((folder + "/LeadJet/p2dRingObservableDeltaPhiVsLeadJetPt").c_str(), "<#it{R}> vs #Delta#varphi_{jet} vs Lead Jet #it{p}_{T};#Delta#varphi_{jet};#it{p}_{T}^{LeadJet} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisJetPt});
      histosRingFamily.add((folder + "/LeadJet/p2dRingObservableDeltaThetaVsLeadJetPt").c_str(), "<#it{R}> vs #Delta#theta_{jet} vs Lead Jet #it{p}_{T};#Delta#theta_{jet};#it{p}_{T}^{LeadJet} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisJetPt});

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
      histosRingFamily.add((folder + "/LeadP/pRingObservableLeadPMass").c_str(), "<#it{R}> vs Mass (LeadP);m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
      histosRingFamily.add((folder + "/SubJet/pRingObservable2ndJetMass").c_str(), "<#it{R}> vs Mass (SubJet);m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile, {axisConfigurations.axisLambdaMassSigExtract});

      // KappaEff moments, per proxy and vs mass for signal extraction:
      for (const std::string& proxy : {std::string("LeadJet"), std::string("LeadP"), std::string("SubJet")}) {
        histosRingFamily.add((folder + "/" + proxy + "/KappaEff/pKappaNum" + proxy + "VsMass").c_str(), ("<u^{2}> vs Mass (" + proxy + ");m_{p#pi} (GeV/c^{2});<u^{2}>").c_str(), kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
        histosRingFamily.add((folder + "/" + proxy + "/KappaEff/pKappaDen" + proxy + "VsMass").c_str(), ("<w> vs Mass (" + proxy + ");m_{p#pi} (GeV/c^{2});<w>").c_str(), kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
        histosRingFamily.add((folder + "/" + proxy + "/KappaEff/pKappaNumTimesDen" + proxy + "VsMass").c_str(), ("<u^{2} w> vs Mass (" + proxy + ");m_{p#pi} (GeV/c^{2});<u^{2} w>").c_str(), kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
        histosRingFamily.add((folder + "/" + proxy + "/KappaEff/pRingTimesDen" + proxy + "VsMass").c_str(), ("<#it{R} w> vs Mass (" + proxy + ");m_{p#pi} (GeV/c^{2});<#it{R} w>").c_str(), kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
        // The same moments per CheapSigExtract region, from v0InMassWindow and v0InMassPeak: bin 1 sideband, bin 2 peak, underflow otherwise
        histosRingFamily.add((folder + "/" + proxy + "/KappaEff/pRing" + proxy + "VsMassRegion").c_str(), ("<#it{R}> per mass region (" + proxy + ");mass region (1 sideband, 2 peak);<#it{R}>").c_str(), kTProfile, {{2, 0., 2.}});
        histosRingFamily.add((folder + "/" + proxy + "/KappaEff/pKappaNum" + proxy + "VsMassRegion").c_str(), ("<u^{2}> per mass region (" + proxy + ");mass region (1 sideband, 2 peak);<u^{2}>").c_str(), kTProfile, {{2, 0., 2.}});
        histosRingFamily.add((folder + "/" + proxy + "/KappaEff/pKappaDen" + proxy + "VsMassRegion").c_str(), ("<w> per mass region (" + proxy + ");mass region (1 sideband, 2 peak);<w>").c_str(), kTProfile, {{2, 0., 2.}});
        histosRingFamily.add((folder + "/" + proxy + "/KappaEff/pKappaNumTimesDen" + proxy + "VsMassRegion").c_str(), ("<u^{2} w> per mass region (" + proxy + ");mass region (1 sideband, 2 peak);<u^{2} w>").c_str(), kTProfile, {{2, 0., 2.}});
      }
      // TProfile2D: <R> vs Mass (DeltaPhi)
      histosRingFamily.add((folder + "/LeadJet/p2dRingObservableDeltaPhiVsMass").c_str(), "<#it{R}> vs #Delta#varphi_{jet} vs Mass;#Delta#varphi_{jet};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisLambdaMassSigExtract});
      // TProfile2D: <R> vs Mass (DeltaTheta)
      histosRingFamily.add((folder + "/LeadJet/p2dRingObservableDeltaThetaVsMass").c_str(), "<#it{R}> vs #Delta#theta_{jet} vs Mass;#Delta#theta_{jet};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisLambdaMassSigExtract});
      // TProfile2D: <R> vs Mass (EtaLambda):
      histosRingFamily.add((folder + "/LeadJet/p2dRingObservableEtaLambdaVsMass").c_str(), "<#it{R}> vs #eta_{#Lambda} vs Mass;#eta_{#Lambda};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisLambdaMassSigExtract});
      // TProfile2D: <R> vs EtaProxy Vs Mass
      histosRingFamily.add((folder + "/LeadJet/p2dRingObservableEtaLeadJetVsMass").c_str(), "<#it{R}> vs #eta_{jet} vs Mass;#eta_{jet};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
      histosRingFamily.add((folder + "/LeadP/p2dRingObservableLeadPEtaLeadPVsMass").c_str(), "<#it{R}> vs #eta_{LeadP} vs Mass;#eta_{LeadP};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
      histosRingFamily.add((folder + "/SubJet/p2dRingObservable2ndJetEta2ndJetVsMass").c_str(), "<#it{R}> vs #eta_{SubJet} vs Mass;#eta_{SubJet};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
      // Auxiliary counters for EtaProxy Vs Mass:
      histosRingFamily.add((folder + "/LeadJet/h2dCounterEtaLeadJetVsMass").c_str(), "V0 Counter vs #eta_{jet} vs Mass;#eta_{jet};m_{p#pi} (GeV/c^{2});Counter", kTH2D, {axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
      histosRingFamily.add((folder + "/LeadP/h2dCounterLeadPEtaLeadPVsMass").c_str(), "V0 Counter vs #eta_{LeadP} vs Mass;#eta_{LeadP};m_{p#pi} (GeV/c^{2});Counter", kTH2D, {axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
      histosRingFamily.add((folder + "/SubJet/h2dCounter2ndJetEta2ndJetVsMass").c_str(), "V0 Counter vs #eta_{SubJet} vs Mass;#eta_{SubJet};m_{p#pi} (GeV/c^{2});Counter", kTH2D, {axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});

      // TProfile2D: <R> vs Eta Lambda vs Eta Jet (Understanding eta dependence seen in pRingEtaCuts)
      histosRingFamily.add((folder + "/LeadJet/EtaDependence/hCounterEtaLambdaMinusEtaJet").c_str(), "N_{V0s} vs #eta_{#Lambda} - #eta_{Jet};#eta_{#Lambda} - #eta_{Jet}; N_{V0s}", kTH1D, {axisConfigurations.axisDeltaEtaCoarse});
      histosRingFamily.add((folder + "/LeadJet/EtaDependence/pRingObservableEtaLambdaMinusEtaJet").c_str(), "<#it{R}> vs #eta_{#Lambda} - #eta_{Jet};#eta_{#Lambda} - #eta_{Jet};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaEtaCoarse});
      histosRingFamily.add((folder + "/LeadJet/EtaDependence/p2dRingObservableEtaLambdaVsEtaJet").c_str(), "<#it{R}> vs #eta_{#Lambda} vs #eta_{Jet};#eta_{#Lambda};#eta_{Jet};<#it{R}>", kTProfile2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisEtaCoarse});
      histosRingFamily.add((folder + "/LeadJet/EtaDependence/p2dRingObservableEtaLambdaVsEtaJet_FineBins").c_str(), "<#it{R}> vs #eta_{#Lambda} vs #eta_{Jet} (fine bins);#eta_{#Lambda};#eta_{Jet};<#it{R}>", kTProfile2D, {axisConfigurations.axisV0Eta, axisConfigurations.axisEta});
      histosRingFamily.add((folder + "/LeadP/EtaDependence/p2dRingObservableEtaLambdaVsEtaLeadP").c_str(), "<#it{R}> vs #eta_{#Lambda} vs #eta_{LeadP};#eta_{#Lambda};#eta_{LeadP};<#it{R}>", kTProfile2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisEtaCoarse});
      histosRingFamily.add((folder + "/SubJet/EtaDependence/p2dRingObservableEtaLambdaVsEta2ndJet").c_str(), "<#it{R}> vs #eta_{#Lambda} vs #eta_{SubJet};#eta_{#Lambda};#eta_{SubJet};<#it{R}>", kTProfile2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisEtaCoarse});
      // Counters for these histograms, instead of only TProfile2Ds:
      histosRingFamily.add((folder + "/LeadJet/EtaDependence/h2dCounterEtaLambdaVsEtaJet").c_str(), "Counts, #eta_{#Lambda} vs #eta_{Jet};#eta_{#Lambda};#eta_{Jet};Counts", kTH2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisEtaCoarse});
      histosRingFamily.add((folder + "/LeadJet/EtaDependence/h2dCounterEtaLambdaVsEtaJet_FineBins").c_str(), "Counts (fine bins), #eta_{#Lambda} vs #eta_{Jet};#eta_{#Lambda};#eta_{Jet};Counts", kTH2D, {axisConfigurations.axisV0Eta, axisConfigurations.axisEta});
      histosRingFamily.add((folder + "/LeadP/EtaDependence/h2dCounterEtaLambdaVsEtaLeadP").c_str(), "Counts, #eta_{#Lambda} vs #eta_{LeadP};#eta_{#Lambda};#eta_{LeadP};Counts", kTH2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisEtaCoarse});
      histosRingFamily.add((folder + "/SubJet/EtaDependence/h2dCounterEtaLambdaVsEta2ndJet").c_str(), "Counts, #eta_{#Lambda} vs #eta_{SubJet};#eta_{#Lambda};#eta_{SubJet};Counts", kTH2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisEtaCoarse});
      // --- TProfile3D: <R> vs DeltaPhi vs Mass vs LambdaPt ---
      histosRingFamily.add((folder + "/LeadJet/p3dRingObservableDeltaPhiVsMassVsLambdaPt").c_str(), "<#it{R}> vs #Delta#varphi_{jet} vs Mass vs #it{p}_{T}^{#Lambda};#Delta#varphi_{jet};m_{p#pi} (GeV/c^{2});#it{p}_{T}^{#Lambda} (GeV/c)", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisPtSigExtract});
      // --- TProfile3D: <R> vs DeltaTheta vs Mass vs LambdaPt ---
      histosRingFamily.add((folder + "/LeadJet/p3dRingObservableDeltaThetaVsMassVsLambdaPt").c_str(), "<#it{R}> vs #Delta#theta_{jet} vs Mass vs #it{p}_{T}^{#Lambda};#Delta#theta_{jet};m_{p#pi} (GeV/c^{2});#it{p}_{T}^{#Lambda} (GeV/c)", kTProfile3D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisPtSigExtract});
      // --- TProfile3D: <R> vs DeltaPhi vs Mass vs LeadJetPt ---
      histosRingFamily.add((folder + "/LeadJet/p3dRingObservableDeltaPhiVsMassVsLeadJetPt").c_str(), "<#it{R}> vs #Delta#varphi_{jet} vs Mass vs Lead Jet #it{p}_{T};#Delta#varphi_{jet};m_{p#pi} (GeV/c^{2});#it{p}_{T}^{LeadJet} (GeV/c)", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisJetPtSigExtract});
      // --- TProfile3D: <R> vs DeltaTheta vs Mass vs LeadJetPt ---
      histosRingFamily.add((folder + "/LeadJet/p3dRingObservableDeltaThetaVsMassVsLeadJetPt").c_str(), "<#it{R}> vs #Delta#theta_{jet} vs Mass vs Lead Jet #it{p}_{T};#Delta#theta_{jet};m_{p#pi} (GeV/c^{2});#it{p}_{T}^{LeadJet} (GeV/c)", kTProfile3D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisJetPtSigExtract});

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
    if (useRingZ) {
      histos.add("RzDiagnostics/pRingVsChiLeadJet", "<#it{R}_{z}> vs #chi (LeadJet) (#hat{n} = cos(#chi)#hat{#phi} + sin(#chi)#hat{#theta});#chi_{LeadJet};<#it{R}_{z}>", kTProfile, {axisConfigurations.axisDeltaPhi}); // chi is the angle such that n_hat = cos(chi) phi_hat + sin(chi) theta_hat.
      histos.add("RzDiagnostics/pRingVsChiLeadP", "<#it{R}_{z}> vs #chi (LeadP) (#hat{n} = cos(#chi)#hat{#phi} + sin(#chi)#hat{#theta});#chi_{LeadP};<#it{R}_{z}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("RzDiagnostics/pRingVsChiSubJet", "<#it{R}_{z}> vs #chi (SubJet) (#hat{n} = cos(#chi)#hat{#phi} + sin(#chi)#hat{#theta});#chi_{SubJet};<#it{R}_{z}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("RzDiagnostics/p2dRingVsChiVsMassLeadJet", "<#it{R}_{z}> vs #chi vs Mass (LeadJet) (#hat{n} = cos(#chi)#hat{#phi} + sin(#chi)#hat{#theta});#chi_{LeadJet};m_{p#pi} (GeV/c^{2});<#it{R}_{z}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("RzDiagnostics/p2dRingVsChiVsMassLeadP", "<#it{R}_{z}> vs #chi vs Mass (LeadP) (#hat{n} = cos(#chi)#hat{#phi} + sin(#chi)#hat{#theta});#chi_{LeadP};m_{p#pi} (GeV/c^{2});<#it{R}_{z}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("RzDiagnostics/p2dRingVsChiVsMassSubJet", "<#it{R}_{z}> vs #chi vs Mass (SubJet) (#hat{n} = cos(#chi)#hat{#phi} + sin(#chi)#hat{#theta});#chi_{SubJet};m_{p#pi} (GeV/c^{2});<#it{R}_{z}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("RzDiagnostics/p2dRingVsDeltaPhiVsDeltaEtaLeadJet", "<#it{R}_{z}> vs (#Delta#varphi_{jet}, #Delta#eta_{jet});#Delta#varphi_{jet}=#phi_{#Lambda} - #phi_{Jet};#eta_{#Lambda}-#eta_{Jet};<#it{R}_{z}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisDeltaEtaCoarse});
      // R_perp = R - R_z, filled directly to store the proper error propagation (R and R_z share every candidate, thus are correlated):
      histos.add("RzDiagnostics/pRingPerpIntegrated", "<#it{R}_{#perp}> per proxy; ;<#it{R}_{#perp}> = <#it{R}> - <#it{R}_{z}>", kTProfile, {{3, 0, 3}});
      histos.get<TProfile>(HIST("RzDiagnostics/pRingPerpIntegrated"))->GetXaxis()->SetBinLabel(1, "LeadJet");
      histos.get<TProfile>(HIST("RzDiagnostics/pRingPerpIntegrated"))->GetXaxis()->SetBinLabel(2, "LeadP");
      histos.get<TProfile>(HIST("RzDiagnostics/pRingPerpIntegrated"))->GetXaxis()->SetBinLabel(3, "SubJet");
      histos.add("RzDiagnostics/pRingPerpLeadJetVsMass", "<#it{R}_{#perp}> vs Mass (LeadJet);m_{p#pi} (GeV/c^{2});<#it{R}_{#perp}> = <#it{R}> - <#it{R}_{z}>", kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
      histos.add("RzDiagnostics/pRingPerpLeadPVsMass", "<#it{R}_{#perp}> vs Mass (LeadP);m_{p#pi} (GeV/c^{2});<#it{R}_{#perp}> = <#it{R}> - <#it{R}_{z}>", kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
      histos.add("RzDiagnostics/pRingPerpSubJetVsMass", "<#it{R}_{#perp}> vs Mass (SubJet);m_{p#pi} (GeV/c^{2});<#it{R}_{#perp}> = <#it{R}> - <#it{R}_{z}>", kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
      histos.add("RzDiagnostics/pRingPerpLeadJetVsPhiAEE", "<#it{R}_{#perp}> vs AEE angle (LeadJet);#phi_{#Lambda-like}-#phi_{p-like}^{*};<#it{R}_{#perp}> = <#it{R}> - <#it{R}_{z}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("RzDiagnostics/pRingPerpLeadPVsPhiAEE", "<#it{R}_{#perp}> vs AEE angle (LeadP);#phi_{#Lambda-like}-#phi_{p-like}^{*};<#it{R}_{#perp}> = <#it{R}> - <#it{R}_{z}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("RzDiagnostics/pRingPerpSubJetVsPhiAEE", "<#it{R}_{#perp}> vs AEE angle (SubJet);#phi_{#Lambda-like}-#phi_{p-like}^{*};<#it{R}_{#perp}> = <#it{R}> - <#it{R}_{z}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("RzDiagnostics/pRingPerpLeadJetVsDeltaPhi", "<#it{R}_{#perp}> vs #Delta#varphi (LeadJet);#Delta#varphi_{jet};<#it{R}_{#perp}> = <#it{R}> - <#it{R}_{z}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("RzDiagnostics/pRingPerpLeadPVsDeltaPhi", "<#it{R}_{#perp}> vs #Delta#varphi (LeadP);#Delta#varphi_{LeadP};<#it{R}_{#perp}> = <#it{R}> - <#it{R}_{z}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("RzDiagnostics/pRingPerpSubJetVsDeltaPhi", "<#it{R}_{#perp}> vs #Delta#varphi (SubJet);#Delta#varphi_{SubJet};<#it{R}_{#perp}> = <#it{R}> - <#it{R}_{z}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      // Standard (unweighted) longitudinal acceptance factor, on the LeadJet sample, to compare against the n_z^2-weighted kappa_z:
      histos.add("RzDiagnostics/pCosSqThetaStarZLeadJetVsMass", "<cos^{2}#theta^{*}_{z}> vs Mass (LeadJet);m_{p#pi} (GeV/c^{2});<cos^{2}#theta^{*}_{z}>", kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
    }

    // Kinematic QA histograms for diagnosing detector defficiencies (V0s and daughters only. Jets are under JetKinematicsQA):
    histos.add("KinematicsQA/V0/hV0Phi", "hV0Phi; #phi_{V0}; Counts", kTH1D, {axisConfigurations.axisPhi});
    histos.add("KinematicsQA/V0/hV0Eta", "hV0Eta; #eta_{V0}; Counts", kTH1D, {axisConfigurations.axisV0Eta});
    histos.add("KinematicsQA/V0/hV0Pt", "hV0Pt; p_{T, V0}; Counts", kTH1D, {axisConfigurations.axisPt});
    histos.add("KinematicsQA/V0/h2dV0PhiVsEta", "h2dV0PhiVsEta; #phi_{V0}; #eta_{V0}; Counts", kTH2D, {axisConfigurations.axisPhi, axisConfigurations.axisV0Eta});
    histos.add("KinematicsQA/V0/h2dV0PhiVsJetPhi", "h2dV0PhiVsJetPhi; #phi_{V0}; #phi_{Jet}; Counts", kTH2D, {axisConfigurations.axisPhi, axisConfigurations.axisPhi});
    histos.add("KinematicsQA/V0/h2dV0EtaVsJetEta", "h2dV0EtaVsJetEta; #eta_{V0}; #eta_{Jet}; Counts", kTH2D, {axisConfigurations.axisV0Eta, axisConfigurations.axisV0Eta});
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

    histos.add("IntegratedCuts/pRingCuts", "pRingCuts; ;<#it{R}>", kTProfile, {{4, 0, 4}});
    histos.get<TProfile>(HIST("IntegratedCuts/pRingCuts"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
    histos.get<TProfile>(HIST("IntegratedCuts/pRingCuts"))->GetXaxis()->SetBinLabel(2, "p_{T}^{#Lambda}@[0.5,1.5],|y_{#Lambda}|<0.5"); // (v0pt > 0.5 && v0pt < 1.5) && std::abs(lambdaRapidity) < 0.5;
    histos.get<TProfile>(HIST("IntegratedCuts/pRingCuts"))->GetXaxis()->SetBinLabel(3, "|Jet_{#eta}|<0.5");
    histos.get<TProfile>(HIST("IntegratedCuts/pRingCuts"))->GetXaxis()->SetBinLabel(4, "#Lambda + Jet cuts");

    // Same for subleading jet and leading particle:
    histos.add("IntegratedCuts/pRingCutsSubLeadingJet", "pRingCutsSubLeadingJet; ;<#it{R}>", kTProfile, {{4, 0, 4}});
    histos.get<TProfile>(HIST("IntegratedCuts/pRingCutsSubLeadingJet"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
    histos.get<TProfile>(HIST("IntegratedCuts/pRingCutsSubLeadingJet"))->GetXaxis()->SetBinLabel(2, "p_{T,#Lambda}@[0.5,1.5],|y_{#Lambda}|<0.5");
    histos.get<TProfile>(HIST("IntegratedCuts/pRingCutsSubLeadingJet"))->GetXaxis()->SetBinLabel(3, "|SubJet_{#eta}|<0.5");
    histos.get<TProfile>(HIST("IntegratedCuts/pRingCutsSubLeadingJet"))->GetXaxis()->SetBinLabel(4, "#Lambda + SubJet cuts");

    histos.add("IntegratedCuts/pRingCutsLeadingP", "pRingCutsLeadingP; ;<#it{R}>", kTProfile, {{4, 0, 4}});
    histos.get<TProfile>(HIST("IntegratedCuts/pRingCutsLeadingP"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
    histos.get<TProfile>(HIST("IntegratedCuts/pRingCutsLeadingP"))->GetXaxis()->SetBinLabel(2, "p_{T}^{#Lambda}@[0.5,1.5],|y_{#Lambda}|<0.5");
    histos.get<TProfile>(HIST("IntegratedCuts/pRingCutsLeadingP"))->GetXaxis()->SetBinLabel(3, "|LeadP_{#eta}|<0.5");
    histos.get<TProfile>(HIST("IntegratedCuts/pRingCutsLeadingP"))->GetXaxis()->SetBinLabel(4, "#Lambda + LeadP cuts");

    // Mass-selected (not properly signal-extracted yet) TProfiles:
    histos.add("IntegratedCuts/p2dRingCutsV0MassPeak", "p2dRingCuts V0MassPeak; ; SidebandWindow (0) or InPeak (1);<#it{R}>", kTProfile2D, {{4, 0, 4}, {2, 0, 2}});
    histos.get<TProfile2D>(HIST("IntegratedCuts/p2dRingCutsV0MassPeak"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
    histos.get<TProfile2D>(HIST("IntegratedCuts/p2dRingCutsV0MassPeak"))->GetXaxis()->SetBinLabel(2, "p_{T}^{#Lambda}@[0.5,1.5],|y_{#Lambda}|<0.5"); // (v0pt > 0.5 && v0pt < 1.5) && std::abs(lambdaRapidity) < 0.5;
    histos.get<TProfile2D>(HIST("IntegratedCuts/p2dRingCutsV0MassPeak"))->GetXaxis()->SetBinLabel(3, "|Jet_{#eta}|<0.5");
    histos.get<TProfile2D>(HIST("IntegratedCuts/p2dRingCutsV0MassPeak"))->GetXaxis()->SetBinLabel(4, "#Lambda + Jet cuts");

    // Same for subleading jet and leading particle:
    histos.add("IntegratedCuts/p2dRingCutsSubLeadingJetV0MassPeak", "p2dRingCutsSubLeadingJet V0MassPeak; ; SidebandWindow (0) or InPeak (1);<#it{R}>", kTProfile2D, {{4, 0, 4}, {2, 0, 2}});
    histos.get<TProfile2D>(HIST("IntegratedCuts/p2dRingCutsSubLeadingJetV0MassPeak"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
    histos.get<TProfile2D>(HIST("IntegratedCuts/p2dRingCutsSubLeadingJetV0MassPeak"))->GetXaxis()->SetBinLabel(2, "p_{T,#Lambda}@[0.5,1.5],|y_{#Lambda}|<0.5");
    histos.get<TProfile2D>(HIST("IntegratedCuts/p2dRingCutsSubLeadingJetV0MassPeak"))->GetXaxis()->SetBinLabel(3, "|SubJet_{#eta}|<0.5");
    histos.get<TProfile2D>(HIST("IntegratedCuts/p2dRingCutsSubLeadingJetV0MassPeak"))->GetXaxis()->SetBinLabel(4, "#Lambda + SubJet cuts");

    histos.add("IntegratedCuts/p2dRingCutsLeadingPV0MassPeak", "p2dRingCutsLeadingP V0MassPeak; ; SidebandWindow (0) or InPeak (1);<#it{R}>", kTProfile2D, {{4, 0, 4}, {2, 0, 2}});
    histos.get<TProfile2D>(HIST("IntegratedCuts/p2dRingCutsLeadingPV0MassPeak"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
    histos.get<TProfile2D>(HIST("IntegratedCuts/p2dRingCutsLeadingPV0MassPeak"))->GetXaxis()->SetBinLabel(2, "p_{T}^{#Lambda}@[0.5,1.5],|y_{#Lambda}|<0.5");
    histos.get<TProfile2D>(HIST("IntegratedCuts/p2dRingCutsLeadingPV0MassPeak"))->GetXaxis()->SetBinLabel(3, "|LeadP_{#eta}|<0.5");
    histos.get<TProfile2D>(HIST("IntegratedCuts/p2dRingCutsLeadingPV0MassPeak"))->GetXaxis()->SetBinLabel(4, "#Lambda + LeadP cuts");

    // Counters for each case to understand statistics loss:
    histos.add("IntegratedCuts/hCountCuts", "hCountCuts; ;N V0s", kTH1D, {{4, 0, 4}});
    histos.get<TH1>(HIST("IntegratedCuts/hCountCuts"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
    histos.get<TH1>(HIST("IntegratedCuts/hCountCuts"))->GetXaxis()->SetBinLabel(2, "p_{T}^{#Lambda}@[0.5,1.5],|y_{#Lambda}|<0.5"); // (v0pt > 0.5 && v0pt < 1.5) && std::abs(lambdaRapidity) < 0.5;
    histos.get<TH1>(HIST("IntegratedCuts/hCountCuts"))->GetXaxis()->SetBinLabel(3, "|Jet_{#eta}|<0.5");
    histos.get<TH1>(HIST("IntegratedCuts/hCountCuts"))->GetXaxis()->SetBinLabel(4, "#Lambda + Jet cuts");

    // Same for subleading jet and leading particle:
    histos.add("IntegratedCuts/hCountCutsSubLeadingJet", "hCountCutsSubLeadingJet; ;N V0s", kTH1D, {{4, 0, 4}});
    histos.get<TH1>(HIST("IntegratedCuts/hCountCutsSubLeadingJet"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
    histos.get<TH1>(HIST("IntegratedCuts/hCountCutsSubLeadingJet"))->GetXaxis()->SetBinLabel(2, "p_{T,#Lambda}@[0.5,1.5],|y_{#Lambda}|<0.5");
    histos.get<TH1>(HIST("IntegratedCuts/hCountCutsSubLeadingJet"))->GetXaxis()->SetBinLabel(3, "|SubJet_{#eta}|<0.5");
    histos.get<TH1>(HIST("IntegratedCuts/hCountCutsSubLeadingJet"))->GetXaxis()->SetBinLabel(4, "#Lambda + SubJet cuts");

    histos.add("IntegratedCuts/hCountCutsLeadingP", "hCountCutsLeadingP; ;N V0s", kTH1D, {{4, 0, 4}});
    histos.get<TH1>(HIST("IntegratedCuts/hCountCutsLeadingP"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
    histos.get<TH1>(HIST("IntegratedCuts/hCountCutsLeadingP"))->GetXaxis()->SetBinLabel(2, "p_{T}^{#Lambda}@[0.5,1.5],|y_{#Lambda}|<0.5");
    histos.get<TH1>(HIST("IntegratedCuts/hCountCutsLeadingP"))->GetXaxis()->SetBinLabel(3, "|LeadP_{#eta}|<0.5");
    histos.get<TH1>(HIST("IntegratedCuts/hCountCutsLeadingP"))->GetXaxis()->SetBinLabel(4, "#Lambda + LeadP cuts");

    // Fake-polarization diagnostics:
    if (qaSwitches.doFakePolDiagnosticsQA) {
      // Integrated observable dependent on jet proxy #eta to unfold possible asymmetries in detector:
      histos.add("EtaStudy/pRingEtaCuts", "pRingEtaCuts; ;<#it{R}>", kTProfile, {{15, 0, 15}});
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCuts"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCuts"))->GetXaxis()->SetBinLabel(2, "#eta_{Jet} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCuts"))->GetXaxis()->SetBinLabel(3, "#eta_{Jet} < 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCuts"))->GetXaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCuts"))->GetXaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCuts"))->GetXaxis()->SetBinLabel(6, "#eta_{Jet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCuts"))->GetXaxis()->SetBinLabel(7, "#eta_{Jet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCuts"))->GetXaxis()->SetBinLabel(8, "#eta_{Jet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCuts"))->GetXaxis()->SetBinLabel(9, "#eta_{Jet} < 0, #eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCuts"))->GetXaxis()->SetBinLabel(10, "#eta_{Jet} > R");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCuts"))->GetXaxis()->SetBinLabel(11, "#eta_{Jet} < -R");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCuts"))->GetXaxis()->SetBinLabel(12, "#eta_{Jet} > R, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCuts"))->GetXaxis()->SetBinLabel(13, "#eta_{Jet} > R, #eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCuts"))->GetXaxis()->SetBinLabel(14, "#eta_{Jet} < -R, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCuts"))->GetXaxis()->SetBinLabel(15, "#eta_{Jet} < -R, #eta_{#Lambda} < 0");

      histos.add("EtaStudy/pRingEtaCutsSubLeadingJet", "pRingEtaCutsSubLeadingJet; ;<#it{R}>", kTProfile, {{15, 0, 15}});
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCutsSubLeadingJet"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCutsSubLeadingJet"))->GetXaxis()->SetBinLabel(2, "#eta_{SubJet} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCutsSubLeadingJet"))->GetXaxis()->SetBinLabel(3, "#eta_{SubJet} < 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCutsSubLeadingJet"))->GetXaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCutsSubLeadingJet"))->GetXaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCutsSubLeadingJet"))->GetXaxis()->SetBinLabel(6, "#eta_{SubJet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCutsSubLeadingJet"))->GetXaxis()->SetBinLabel(7, "#eta_{SubJet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCutsSubLeadingJet"))->GetXaxis()->SetBinLabel(8, "#eta_{SubJet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCutsSubLeadingJet"))->GetXaxis()->SetBinLabel(9, "#eta_{SubJet} < 0, #eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCutsSubLeadingJet"))->GetXaxis()->SetBinLabel(10, "#eta_{SubJet} > R");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCutsSubLeadingJet"))->GetXaxis()->SetBinLabel(11, "#eta_{SubJet} < -R");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCutsSubLeadingJet"))->GetXaxis()->SetBinLabel(12, "#eta_{SubJet} > R, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCutsSubLeadingJet"))->GetXaxis()->SetBinLabel(13, "#eta_{SubJet} > R, #eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCutsSubLeadingJet"))->GetXaxis()->SetBinLabel(14, "#eta_{SubJet} < -R, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCutsSubLeadingJet"))->GetXaxis()->SetBinLabel(15, "#eta_{SubJet} < -R, #eta_{#Lambda} < 0");

      histos.add("EtaStudy/pRingEtaCutsLeadingP", "pRingEtaCutsLeadingP; ;<#it{R}>", kTProfile, {{9, 0, 9}});
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCutsLeadingP"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCutsLeadingP"))->GetXaxis()->SetBinLabel(2, "#eta_{LeadP} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCutsLeadingP"))->GetXaxis()->SetBinLabel(3, "#eta_{LeadP} < 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCutsLeadingP"))->GetXaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCutsLeadingP"))->GetXaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCutsLeadingP"))->GetXaxis()->SetBinLabel(6, "#eta_{LeadP} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCutsLeadingP"))->GetXaxis()->SetBinLabel(7, "#eta_{LeadP} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCutsLeadingP"))->GetXaxis()->SetBinLabel(8, "#eta_{LeadP} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("EtaStudy/pRingEtaCutsLeadingP"))->GetXaxis()->SetBinLabel(9, "#eta_{LeadP} < 0, #eta_{#Lambda} < 0");

      // Studying the signal Vs background integral (a very naive estimative of the invariant mass peak)
      histos.add("EtaStudy/pRingEtaCutsLeadingP_MassSignalVsBackground", "pRingEtaCutsLeadingP_MassSignalVsBackground; ; ;<#it{R}>", kTProfile2D, {{9, 0, 9}, {2, 0, 2}});
      histos.get<TProfile2D>(HIST("EtaStudy/pRingEtaCutsLeadingP_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TProfile2D>(HIST("EtaStudy/pRingEtaCutsLeadingP_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(2, "#eta_{LeadP} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pRingEtaCutsLeadingP_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(3, "#eta_{LeadP} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pRingEtaCutsLeadingP_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pRingEtaCutsLeadingP_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pRingEtaCutsLeadingP_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(6, "#eta_{LeadP} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pRingEtaCutsLeadingP_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(7, "#eta_{LeadP} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pRingEtaCutsLeadingP_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(8, "#eta_{LeadP} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pRingEtaCutsLeadingP_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(9, "#eta_{LeadP} < 0, #eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pRingEtaCutsLeadingP_MassSignalVsBackground"))->GetYaxis()->SetBinLabel(1, "#Lambda out of mass peak");
      histos.get<TProfile2D>(HIST("EtaStudy/pRingEtaCutsLeadingP_MassSignalVsBackground"))->GetYaxis()->SetBinLabel(2, "#Lambda in mass peak");

      // Fake polarization signal QA
      // --> The "negative helicity problem", where topologies with a proton decaying opposite to the Lambda momentum are enhanced by
      // efficiency of reconstruction. The geometries where the proton moves in the same direction as the boost will have a very small
      // momentum pion, which is not as easily detected as the opposite case! This may insert a fake signal of polarization in the measurement!
      histos.add("EtaStudy/hFakePolCounts", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda};", kTH2D, {axisConfigurations.axisCosTheta, {9, 0, 9}});
      histos.get<TH2>(HIST("EtaStudy/hFakePolCounts"))->GetZaxis()->SetTitle("N_{V0s}");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCounts"))->GetYaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCounts"))->GetYaxis()->SetBinLabel(2, "#eta_{Jet} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCounts"))->GetYaxis()->SetBinLabel(3, "#eta_{Jet} < 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCounts"))->GetYaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCounts"))->GetYaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCounts"))->GetYaxis()->SetBinLabel(6, "#eta_{Jet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCounts"))->GetYaxis()->SetBinLabel(7, "#eta_{Jet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCounts"))->GetYaxis()->SetBinLabel(8, "#eta_{Jet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCounts"))->GetYaxis()->SetBinLabel(9, "#eta_{Jet} < 0, #eta_{#Lambda} < 0");

      // The same, but for actual signal instead of counts:
      histos.add("EtaStudy/pFakePolSignalVsCosTheta", "FakePolSignal; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda};", kTProfile2D, {axisConfigurations.axisCosTheta, {9, 0, 9}});
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalVsCosTheta"))->GetZaxis()->SetTitle("<#it{R}>");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(2, "#eta_{Jet} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(3, "#eta_{Jet} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(6, "#eta_{Jet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(7, "#eta_{Jet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(8, "#eta_{Jet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(9, "#eta_{Jet} < 0, #eta_{#Lambda} < 0");

      // Seeing the dependence between phi* = atan2(p_p_star \cdot (p_Lambda_hat \times (z_hat \cross p_Lambda)), p_p_star \cdot (z_hat \cross p_Lambda))
      // e_z = p_Lambda_hat; // e_x = normalize(z_hat cross p_Lambda); // e_y = e_z cross e_x;
      // phi_star = atan2(p_p_star dot e_y, p_p_star dot e_x);
      histos.add("HelicityEfficiencyQA/hFakePolCounts_CosThetaVsPhiStar", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #phi^{*}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaPhi});
      histos.add("HelicityEfficiencyQA/pFakePolSignal_CosThetaVsPhiStar", "FakePolSignal; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #phi^{*}", kTProfile2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaPhi});
      // Specific counter for when we have leading jets (relates directly to pFakePolSignal_CosThetaVsPhiStar):
      histos.add("HelicityEfficiencyQA/hFakePolCountsJet_CosThetaVsPhiStar", "FakePolCounts - HasValidLeadJet OK; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #phi^{*}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaPhi});

      // Similar split, but for AEE instead of HEE:
      histos.add("EtaStudy/hCountsVsPhiStar", "FakePolCounts, AEE dependence; #phi^{*};", kTH2D, {axisConfigurations.axisDeltaPhi, {9, 0, 9}});
      histos.get<TH2>(HIST("EtaStudy/hCountsVsPhiStar"))->GetZaxis()->SetTitle("Counts");
      histos.get<TH2>(HIST("EtaStudy/hCountsVsPhiStar"))->GetYaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TH2>(HIST("EtaStudy/hCountsVsPhiStar"))->GetYaxis()->SetBinLabel(2, "#eta_{Jet} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/hCountsVsPhiStar"))->GetYaxis()->SetBinLabel(3, "#eta_{Jet} < 0");
      histos.get<TH2>(HIST("EtaStudy/hCountsVsPhiStar"))->GetYaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/hCountsVsPhiStar"))->GetYaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("EtaStudy/hCountsVsPhiStar"))->GetYaxis()->SetBinLabel(6, "#eta_{Jet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/hCountsVsPhiStar"))->GetYaxis()->SetBinLabel(7, "#eta_{Jet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("EtaStudy/hCountsVsPhiStar"))->GetYaxis()->SetBinLabel(8, "#eta_{Jet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/hCountsVsPhiStar"))->GetYaxis()->SetBinLabel(9, "#eta_{Jet} < 0, #eta_{#Lambda} < 0");
      // For the ring observable as well:
      // Explicitly, checking the Phi* dependence on a series of #eta cuts.
      // The fake, AEE-induced, signal should be zero when integrating on full solid angle, and then have some shape for each eta slice.
      histos.add("EtaStudy/pFakePolSignalvsPhiStar", "FakePolSignal, AEE dependence; #phi^{*};", kTProfile2D, {axisConfigurations.axisDeltaPhi, {9, 0, 9}});
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalvsPhiStar"))->GetZaxis()->SetTitle("<#it{R}>");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalvsPhiStar"))->GetYaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalvsPhiStar"))->GetYaxis()->SetBinLabel(2, "#eta_{Jet} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalvsPhiStar"))->GetYaxis()->SetBinLabel(3, "#eta_{Jet} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalvsPhiStar"))->GetYaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalvsPhiStar"))->GetYaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalvsPhiStar"))->GetYaxis()->SetBinLabel(6, "#eta_{Jet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalvsPhiStar"))->GetYaxis()->SetBinLabel(7, "#eta_{Jet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalvsPhiStar"))->GetYaxis()->SetBinLabel(8, "#eta_{Jet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalvsPhiStar"))->GetYaxis()->SetBinLabel(9, "#eta_{Jet} < 0, #eta_{#Lambda} < 0");

      // For the phi_Lambda - phi_D^* dependency as well:
      histos.add("EtaStudy/pFakePolSignalvsPhiLambdaMinusPhiProtonStar", "FakePolSignal, AEE dependence; #phi_{#Lambda}-#phi_{p}^{*};", kTProfile2D, {axisConfigurations.axisDeltaPhi, {9, 0, 9}});
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalvsPhiLambdaMinusPhiProtonStar"))->GetZaxis()->SetTitle("<#it{R}>");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalvsPhiLambdaMinusPhiProtonStar"))->GetYaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalvsPhiLambdaMinusPhiProtonStar"))->GetYaxis()->SetBinLabel(2, "#eta_{Jet} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalvsPhiLambdaMinusPhiProtonStar"))->GetYaxis()->SetBinLabel(3, "#eta_{Jet} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalvsPhiLambdaMinusPhiProtonStar"))->GetYaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalvsPhiLambdaMinusPhiProtonStar"))->GetYaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalvsPhiLambdaMinusPhiProtonStar"))->GetYaxis()->SetBinLabel(6, "#eta_{Jet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalvsPhiLambdaMinusPhiProtonStar"))->GetYaxis()->SetBinLabel(7, "#eta_{Jet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalvsPhiLambdaMinusPhiProtonStar"))->GetYaxis()->SetBinLabel(8, "#eta_{Jet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("EtaStudy/pFakePolSignalvsPhiLambdaMinusPhiProtonStar"))->GetYaxis()->SetBinLabel(9, "#eta_{Jet} < 0, #eta_{#Lambda} < 0");

      // More about possible AEE dependencies (should see an invariance with JetEta and <R>/jetZ):
      // TODO: think about these error bars: do they still make sense via regular TProfile's SEM error? These are just a quick check, so wouldn't bother much about it.
      histos.add("HelicityEfficiencyQA/pRingVsJetZcomponent", "<#it{R}> vs #hat{t}_{z}; #hat{t}_{z}; <#it{R}>", kTProfile, {axisConfigurations.axisProxyZ}); // Numerically stable and can also show the sign flip (essentially the <#it{R}> vs Eta Jet plot in another scale)
      histos.add("HelicityEfficiencyQA/pRingOverJetZcomponent_VsJetEta", "<#it{R}>/#hat{t}_{z}; #eta_{Jet}; <#it{R}>/#hat{t}_{z}", kTProfile, {axisConfigurations.axisEtaCoarse});
      histos.add("HelicityEfficiencyQA/pRingOverJetZcomponent_VsCosThetaHEE", "<#it{R}>/#hat{t}_{z} Vs cos(#theta) HEE; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; <#it{R}>/#hat{t}_{z}", kTProfile, {axisConfigurations.axisCosTheta});
      histos.add("HelicityEfficiencyQA/pRingOverJetZcomponent_VsPhiStar", "<#it{R}>/#hat{t}_{z} Vs #phi^{*}; #phi^{*} = atan2(#vec{p}^{*}_{p} #cdot [#hat{p}_{#Lambda} #times (#hat{z} #times #hat{p}_{#Lambda})] , #vec{p}^{*}_{p} #cdot (#hat{z} #times #hat{p}_{#Lambda})); <#it{R}>/#hat{t}_{z}", kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("HelicityEfficiencyQA/pRingOverJetZcomponent_VsJetEtaVsCosThetaHEE", "<#it{R}>/#hat{t}_{z}; #eta_{Jet}; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; <#it{R}>/#hat{t}_{z}", kTProfile2D, {axisConfigurations.axisEtaCoarse, axisConfigurations.axisCosTheta});
      histos.add("HelicityEfficiencyQA/pRingOverJetZcomponent_VsJetEtaVsPhiStar", "<#it{R}>/#hat{t}_{z}; #eta_{Jet}; #phi^{*}; <#it{R}>/#hat{t}_{z}", kTProfile2D, {axisConfigurations.axisEtaCoarse, axisConfigurations.axisDeltaPhi});

      // Seeing if phi* is indeed influenced by the DCA between daughters:
      histos.add("HelicityEfficiencyQA/hFakePolCountsJet_PhiStarVsDCAdau", "FakePolCounts - HasValidLeadJet OK; #phi^{*}; DCA_{V0 Daughters}", kTH2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdau});
      histos.add("HelicityEfficiencyQA/pFakePolSignalJet_PhiStarVsDCAdau", "FakePolSignal; #phi^{*}; DCA_{V0 Daughters}", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdau});

      // Adding a way to check if the jet eta is positive or negative as well
      // (AEE signal could be closer to zero otherwise: phi^* dependency may not make it fall to zero as we are no longer integrating <R> in full solid angle, yet analyzing this other dependency is also important)
      histos.add("HelicityEfficiencyQA/pFakePolSignalJet_PhiStarVsDCAdauVsEtaJet", "FakePolSignal; #phi^{*}; DCA_{V0 Daughters}; #eta_{Jet} sign", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdau, {2, -0.9, 0.9}});
      histos.add("HelicityEfficiencyQA/pFakePolSignalJet_PhiStarVsDCAdauVsEtaLambda", "FakePolSignal; #phi^{*}; DCA_{V0 Daughters}; #eta_{#Lambda} sign", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdau, {2, -0.9, 0.9}});

      // Similar checks for dcaPosToPV and dcaNegToPV, which influence AEE the strongest:
      histos.add("HelicityEfficiencyQA/hFakePolCountsJet_PhiStarVsDCAProLike", "FakePolCounts - HasValidLeadJet OK; #phi^{*}; DCA_{PosPV}", kTH2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdauPV});
      histos.add("HelicityEfficiencyQA/pFakePolSignalJet_PhiStarVsDCAProLike", "FakePolSignal; #phi^{*}; DCA_{PosPV}", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdauPV});
      histos.add("HelicityEfficiencyQA/pFakePolSignalJet_PhiStarVsDCAProLikeVsEtaJet", "FakePolSignal; #phi^{*}; DCA_{PosPV}; #eta_{Jet} sign", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdauPV, {2, -0.9, 0.9}});
      histos.add("HelicityEfficiencyQA/pFakePolSignalJet_PhiStarVsDCAProLikeVsEtaLambda", "FakePolSignal; #phi^{*}; DCA_{PosPV}; #eta_{#Lambda} sign", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdauPV, {2, -0.9, 0.9}});

      histos.add("HelicityEfficiencyQA/hFakePolCountsJet_PhiStarVsDCAPiLike", "FakePolCounts - HasValidLeadJet OK; #phi^{*}; DCA_{NegPV}", kTH2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdauPV});
      histos.add("HelicityEfficiencyQA/pFakePolSignalJet_PhiStarVsDCAPiLike", "FakePolSignal; #phi^{*}; DCA_{NegPV}", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdauPV});
      histos.add("HelicityEfficiencyQA/pFakePolSignalJet_PhiStarVsDCAPiLikeVsEtaJet", "FakePolSignal; #phi^{*}; DCA_{NegPV}; #eta_{Jet} sign", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdauPV, {2, -0.9, 0.9}});
      histos.add("HelicityEfficiencyQA/pFakePolSignalJet_PhiStarVsDCAPiLikeVsEtaLambda", "FakePolSignal; #phi^{*}; DCA_{NegPV}; #eta_{#Lambda} sign", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdauPV, {2, -0.9, 0.9}});

      // Doing the same HEE study for leading particles:
      // (eta_{Jet} may be a bad estimator!)
      histos.add("EtaStudy/hFakePolCountsLeadP", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda};", kTH2D, {axisConfigurations.axisCosTheta, {9, 0, 9}});
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLeadP"))->GetZaxis()->SetTitle("N_{V0s}");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLeadP"))->GetYaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLeadP"))->GetYaxis()->SetBinLabel(2, "#eta_{LeadP} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLeadP"))->GetYaxis()->SetBinLabel(3, "#eta_{LeadP} < 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLeadP"))->GetYaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLeadP"))->GetYaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLeadP"))->GetYaxis()->SetBinLabel(6, "#eta_{LeadP} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLeadP"))->GetYaxis()->SetBinLabel(7, "#eta_{LeadP} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLeadP"))->GetYaxis()->SetBinLabel(8, "#eta_{LeadP} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLeadP"))->GetYaxis()->SetBinLabel(9, "#eta_{LeadP} < 0, #eta_{#Lambda} < 0");

      // Avoid fake signal by jets boosting the Lambda in its own direction, then modifying efficiency of reconstruction in a similar way:
      histos.add("EtaStudy/hFakePolCountsLambdaPtCut", "FakePol,p_{T}^{#Lambda}#in[0.5,1.5]; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda};", kTH2D, {axisConfigurations.axisCosTheta, {9, 0, 9}});
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLambdaPtCut"))->GetZaxis()->SetTitle("N_{V0s}");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(2, "#eta_{Jet} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(3, "#eta_{Jet} < 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(6, "#eta_{Jet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(7, "#eta_{Jet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(8, "#eta_{Jet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(9, "#eta_{Jet} < 0, #eta_{#Lambda} < 0");

      // Even stricter cut (also demands rapidity cut stricter than jets, so may see different boosting):
      histos.add("EtaStudy/hFakePolCountsLambdaPtYCuts", "FakePol,p_{T}^{#Lambda}#in[0.5,1.5],|y_{#Lambda}|<0.5; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda};", kTH2D, {axisConfigurations.axisCosTheta, {9, 0, 9}});
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLambdaPtYCuts"))->GetZaxis()->SetTitle("N_{V0s}");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(2, "#eta_{Jet} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(3, "#eta_{Jet} < 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(6, "#eta_{Jet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(7, "#eta_{Jet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(8, "#eta_{Jet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("EtaStudy/hFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(9, "#eta_{Jet} < 0, #eta_{#Lambda} < 0");

      // Another useful quantity -- How much is the fake signal related to the jet's momentum (how much the fake signal is correlated with the Jet-Lambda angular separation):
      histos.add("HelicityEfficiencyQA/hFakePolCountsVsDeltaThetaJet", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #Delta#theta_{Jet}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaTheta});
      histos.add("HelicityEfficiencyQA/hFakePolCountsVsDeltaThetaJetPosEta", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #Delta#theta_{Jet}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaTheta});
      histos.add("HelicityEfficiencyQA/hFakePolCountsVsDeltaThetaJetNegEta", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #Delta#theta_{Jet}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaTheta});
      histos.add("HelicityEfficiencyQA/hFakePolCountsVsDeltaThetaLeadP", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #Delta#theta_{LeadP}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaTheta});
      histos.add("HelicityEfficiencyQA/hFakePolCountsVsDeltaThetaLeadPPosEta", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #Delta#theta_{LeadP}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaTheta});
      histos.add("HelicityEfficiencyQA/hFakePolCountsVsDeltaThetaLeadPNegEta", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #Delta#theta_{LeadP}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaTheta});

      // Understanding the dip at the cos = -1 end:
      histos.add("HelicityEfficiencyQA/hFakePolCountsCosThetaVsPtForJets", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; p_{T}^{#Lambda}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisPtCoarseQA});
      histos.add("HelicityEfficiencyQA/hFakePolCountsCosThetaVsPtForLeadP", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; p_{T}^{#Lambda}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisPtCoarseQA});

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
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/hLambdaMassVsPhiLambdaMinusPhiProtonStar", "m_{#Lambda}, AEE probe; m_{Inv}; #phi_{#Lambda}-#phi_{p}^{*} ; Counts", kTH2D, {axisConfigurations.axisLambdaMass, axisConfigurations.axisDeltaPhi});
      if (analyseAntiLambda)
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/hAntiLambdaMassVsPhiLambdaMinusPhiProtonStar", "m_{#bar{#Lambda}}, AEE probe; m_{Inv}; #phi_{#bar{#Lambda}} - #phi_{p}^{*} ; Counts", kTH2D, {axisConfigurations.axisLambdaMass, axisConfigurations.axisDeltaPhi});

      // The Helicity Efficiency Effect resolved in three mass slices (left sideband, peak, right sideband).
      if (analyseLambda) {
        histos.add("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingLambdaCosThetaHEEVsMass", "<#it{R}>_{LeadJet} vs cos(#theta)_{HEE} vs mass slice, #Lambda;cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisCosThetaCoarse, axisConfigurations.axisLambdaMassThreeBin});
        // Explicit checks to decouple phi-phi* and cosTheta* effects (is the pattern controlled by phi-phi* or by cosTheta*?)
        histos.add("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingLambdaCosThetaHEEVsMassPosDeltaPhiAEE", "<#it{R}>_{LeadJet} (#phi_{#Lambda}-#phi_{p}^{*}>0) vs cos(#theta)_{HEE} vs mass slice, #Lambda;cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisCosThetaCoarse, axisConfigurations.axisLambdaMassThreeBin});
        histos.add("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingLambdaCosThetaHEEVsMassNegDeltaPhiAEE", "<#it{R}>_{LeadJet} (#phi_{#Lambda}-#phi_{p}^{*}<0) vs cos(#theta)_{HEE} vs mass slice, #Lambda;cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisCosThetaCoarse, axisConfigurations.axisLambdaMassThreeBin});
      }
      if (analyseAntiLambda) {
        histos.add("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingAntiLambdaCosThetaHEEVsMass", "<#it{R}>_{LeadJet} vs cos(#theta)_{HEE} vs mass slice, #bar{#Lambda};cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#bar{#Lambda}};m_{#bar{p}#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisCosThetaCoarse, axisConfigurations.axisLambdaMassThreeBin});
        histos.add("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingAntiLambdaCosThetaHEEVsMassPosDeltaPhiAEE", "<#it{R}>_{LeadJet} (#phi_{#Lambda}-#phi_{p}^{*}>0) vs cos(#theta)_{HEE} vs mass slice, #bar{#Lambda};cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#bar{#Lambda}};m_{#bar{p}#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisCosThetaCoarse, axisConfigurations.axisLambdaMassThreeBin});
        histos.add("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingAntiLambdaCosThetaHEEVsMassNegDeltaPhiAEE", "<#it{R}>_{LeadJet} (#phi_{#Lambda}-#phi_{p}^{*}<0) vs cos(#theta)_{HEE} vs mass slice, #bar{#Lambda};cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#bar{#Lambda}};m_{#bar{p}#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisCosThetaCoarse, axisConfigurations.axisLambdaMassThreeBin});
      }
      histos.add("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingAllV0sCosThetaHEEVsMass", "<#it{R}>_{LeadJet} vs cos(#theta)_{HEE} vs mass slice, all V0s;cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda-like};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisCosThetaCoarse, axisConfigurations.axisLambdaMassThreeBin});
      histos.add("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingAllV0sCosThetaHEEVsMassPosDeltaPhiAEE", "<#it{R}>_{LeadJet} (#phi_{#Lambda}-#phi_{p}^{*}>0) vs cos(#theta)_{HEE} vs mass slice, all V0s;cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda-like};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisCosThetaCoarse, axisConfigurations.axisLambdaMassThreeBin});
      histos.add("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingAllV0sCosThetaHEEVsMassNegDeltaPhiAEE", "<#it{R}>_{LeadJet} (#phi_{#Lambda}-#phi_{p}^{*}<0) vs cos(#theta)_{HEE} vs mass slice, all V0s;cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda-like};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisCosThetaCoarse, axisConfigurations.axisLambdaMassThreeBin});

      // <R> Vs PhiLambda-PhiProtonStar dependencies:
      // (here there should be no clear dependency, unless Lambdas or antiLambdas dominate over one another. This inclusive set is QA for the competition)
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/pRingObservableLeadJetVsPhiLambdaLikePhiProtonStar", "<#it{R}>_{LeadJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*};#phi_{#Lambda-like}-#phi_{p-like}^{*};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/pRingObservableLeadPVsPhiLambdaLikePhiProtonStar", "<#it{R}>_{LeadP} vs #phi_{#Lambda-like}-#phi_{p-like}^{*};#phi_{#Lambda-like}-#phi_{p-like}^{*};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/pRingObservable2ndJetVsPhiLambdaLikePhiProtonStar", "<#it{R}>_{SubJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*};#phi_{#Lambda-like}-#phi_{p-like}^{*};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      // Lambda-specific:
      if (analyseLambda) {
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/pRingObservableLeadJetVsPhiLambdaPhiProtonStar", "<#it{R}>_{LeadJet} vs #phi_{#Lambda}-#phi_{p}^{*};#phi_{#Lambda}-#phi_{p}^{*};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaPhi});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/pRingObservableLeadPVsPhiLambdaPhiProtonStar", "<#it{R}>_{LeadP} vs #phi_{#Lambda}-#phi_{p}^{*};#phi_{#Lambda}-#phi_{p}^{*};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaPhi});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/pRingObservable2ndJetVsPhiLambdaPhiProtonStar", "<#it{R}>_{SubJet} vs #phi_{#Lambda}-#phi_{p}^{*};#phi_{#Lambda}-#phi_{p}^{*};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      }
      // Anti-Lambda-specific:
      if (analyseAntiLambda) {
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/pRingObservableLeadJetVsPhiAntiLambdaPhiProtonStar", "<#it{R}>_{LeadJet} vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaPhi});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/pRingObservableLeadPVsPhiAntiLambdaPhiProtonStar", "<#it{R}>_{LeadP} vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaPhi});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/pRingObservable2ndJetVsPhiAntiLambdaPhiProtonStar", "<#it{R}>_{SubJet} vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      }

      // 2D dependencies for signal extraction:
      // (coarser angular axis here: the mass axis takes the statistics, and each cell still has to sustain a fit)
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiLambdaLikePhiProtonStarVsMass", "<#it{R}>_{LeadJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiLambdaLikePhiProtonStarVsMass", "<#it{R}>_{LeadP} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiLambdaLikePhiProtonStarVsMass", "<#it{R}>_{SubJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      // Splitting in proxy eta:
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiLambdaLikePhiProtonStarVsMassPosProxyEta", "<#it{R}>_{LeadJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass, #eta_{Proxy}>0;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiLambdaLikePhiProtonStarVsMassPosProxyEta", "<#it{R}>_{LeadP} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass, #eta_{Proxy}>0;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiLambdaLikePhiProtonStarVsMassPosProxyEta", "<#it{R}>_{SubJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass, #eta_{Proxy}>0;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiLambdaLikePhiProtonStarVsMassNegProxyEta", "<#it{R}>_{LeadJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass, #eta_{Proxy}<0;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiLambdaLikePhiProtonStarVsMassNegProxyEta", "<#it{R}>_{LeadP} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass, #eta_{Proxy}<0;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiLambdaLikePhiProtonStarVsMassNegProxyEta", "<#it{R}>_{SubJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass, #eta_{Proxy}<0;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      // AEE angle Vs Proxy eta dependence:
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiLambdaLikePhiProtonStarVsProxyEtaSign", "<#it{R}>_{LeadJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy} sign;<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, {2, -1, 1}});
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiLambdaLikePhiProtonStarVsProxyEtaSign", "<#it{R}>_{LeadP} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy} sign;<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, {2, -1, 1}});
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiLambdaLikePhiProtonStarVsProxyEtaSign", "<#it{R}>_{SubJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy} sign;<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, {2, -1, 1}});
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiLambdaLikePhiProtonStarVsProxyEta", "<#it{R}>_{LeadJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy};<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEta});
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiLambdaLikePhiProtonStarVsProxyEta", "<#it{R}>_{LeadP} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy};<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEta});
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiLambdaLikePhiProtonStarVsProxyEta", "<#it{R}>_{SubJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy};<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEta});
      // A 3D split that allows for a proper signal extraction after separating the main drivers of fake polarization signal:
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p3dRingObservableLeadJetVsPhiLambdaLikePhiProtonStarVsProxyEtaVsMass", "<#it{R}>_{LeadJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy}; m_{p#pi};<#it{R}>", kTProfile3D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p3dRingObservableLeadPVsPhiLambdaLikePhiProtonStarVsProxyEtaVsMass", "<#it{R}>_{LeadP} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy}; m_{p#pi};<#it{R}>", kTProfile3D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p3dRingObservable2ndJetVsPhiLambdaLikePhiProtonStarVsProxyEtaVsMass", "<#it{R}>_{SubJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy}; m_{p#pi};<#it{R}>", kTProfile3D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
      // Lambda-specific signal extraction:
      if (analyseLambda) {
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiLambdaPhiProtonStarVsMass", "<#it{R}>_{LeadJet} vs #phi_{#Lambda}-#phi_{p}^{*} Vs Mass;#phi_{#Lambda}-#phi_{p}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiLambdaPhiProtonStarVsMass", "<#it{R}>_{LeadP} vs #phi_{#Lambda}-#phi_{p}^{*} Vs Mass;#phi_{#Lambda}-#phi_{p}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiLambdaPhiProtonStarVsMass", "<#it{R}>_{SubJet} vs #phi_{#Lambda}-#phi_{p}^{*} Vs Mass;#phi_{#Lambda}-#phi_{p}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
        // Splitting in proxy eta:
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiLambdaPhiProtonStarVsMassPosProxyEta", "<#it{R}>_{LeadJet} vs #phi_{#Lambda}-#phi_{p}^{*} Vs Mass, #eta_{Proxy}>0;#phi_{#Lambda}-#phi_{p}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiLambdaPhiProtonStarVsMassPosProxyEta", "<#it{R}>_{LeadP} vs #phi_{#Lambda}-#phi_{p}^{*} Vs Mass, #eta_{Proxy}>0;#phi_{#Lambda}-#phi_{p}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiLambdaPhiProtonStarVsMassPosProxyEta", "<#it{R}>_{SubJet} vs #phi_{#Lambda}-#phi_{p}^{*} Vs Mass, #eta_{Proxy}>0;#phi_{#Lambda}-#phi_{p}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiLambdaPhiProtonStarVsMassNegProxyEta", "<#it{R}>_{LeadJet} vs #phi_{#Lambda}-#phi_{p}^{*} Vs Mass, #eta_{Proxy}<0;#phi_{#Lambda}-#phi_{p}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiLambdaPhiProtonStarVsMassNegProxyEta", "<#it{R}>_{LeadP} vs #phi_{#Lambda}-#phi_{p}^{*} Vs Mass, #eta_{Proxy}<0;#phi_{#Lambda}-#phi_{p}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiLambdaPhiProtonStarVsMassNegProxyEta", "<#it{R}>_{SubJet} vs #phi_{#Lambda}-#phi_{p}^{*} Vs Mass, #eta_{Proxy}<0;#phi_{#Lambda}-#phi_{p}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiLambdaPhiProtonStarVsProxyEtaSign", "<#it{R}>_{LeadJet} vs #phi_{#Lambda}-#phi_{p}^{*} Vs #eta_{Proxy};#phi_{#Lambda}-#phi_{p}^{*}; #eta_{Proxy} sign;<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, {2, -1, 1}});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiLambdaPhiProtonStarVsProxyEtaSign", "<#it{R}>_{LeadP} vs #phi_{#Lambda}-#phi_{p}^{*} Vs #eta_{Proxy};#phi_{#Lambda}-#phi_{p}^{*}; #eta_{Proxy} sign;<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, {2, -1, 1}});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiLambdaPhiProtonStarVsProxyEtaSign", "<#it{R}>_{SubJet} vs #phi_{#Lambda}-#phi_{p}^{*} Vs #eta_{Proxy};#phi_{#Lambda}-#phi_{p}^{*}; #eta_{Proxy} sign;<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, {2, -1, 1}});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiLambdaPhiProtonStarVsProxyEta", "<#it{R}>_{LeadJet} vs #phi_{#Lambda}-#phi_{p}^{*} Vs #eta_{Proxy};#phi_{#Lambda}-#phi_{p}^{*}; #eta_{Proxy};<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEta});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiLambdaPhiProtonStarVsProxyEta", "<#it{R}>_{LeadP} vs #phi_{#Lambda}-#phi_{p}^{*} Vs #eta_{Proxy};#phi_{#Lambda}-#phi_{p}^{*}; #eta_{Proxy};<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEta});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiLambdaPhiProtonStarVsProxyEta", "<#it{R}>_{SubJet} vs #phi_{#Lambda}-#phi_{p}^{*} Vs #eta_{Proxy};#phi_{#Lambda}-#phi_{p}^{*}; #eta_{Proxy};<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEta});
        // A 3D split that allows for a proper signal extraction after separating the main drivers of fake polarization signal:
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p3dRingObservableLeadJetVsPhiLambdaPhiProtonStarVsProxyEtaVsMass", "<#it{R}>_{LeadJet} vs #phi_{#Lambda}-#phi_{p}^{*} Vs #eta_{Proxy};#phi_{#Lambda}-#phi_{p}^{*}; #eta_{Proxy}; m_{p#pi};<#it{R}>", kTProfile3D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p3dRingObservableLeadPVsPhiLambdaPhiProtonStarVsProxyEtaVsMass", "<#it{R}>_{LeadP} vs #phi_{#Lambda}-#phi_{p}^{*} Vs #eta_{Proxy};#phi_{#Lambda}-#phi_{p}^{*}; #eta_{Proxy}; m_{p#pi};<#it{R}>", kTProfile3D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p3dRingObservable2ndJetVsPhiLambdaPhiProtonStarVsProxyEtaVsMass", "<#it{R}>_{SubJet} vs #phi_{#Lambda}-#phi_{p}^{*} Vs #eta_{Proxy};#phi_{#Lambda}-#phi_{p}^{*}; #eta_{Proxy}; m_{p#pi};<#it{R}>", kTProfile3D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
      }
      // Anti-Lambda-specific signal extraction:
      if (analyseAntiLambda) {
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiAntiLambdaPhiProtonStarVsMass", "<#it{R}>_{LeadJet} vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} Vs Mass;#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};m_{#bar{p}#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiAntiLambdaPhiProtonStarVsMass", "<#it{R}>_{LeadP} vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} Vs Mass;#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};m_{#bar{p}#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiAntiLambdaPhiProtonStarVsMass", "<#it{R}>_{SubJet} vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} Vs Mass;#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};m_{#bar{p}#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
        // Splitting in proxy eta:
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiAntiLambdaPhiProtonStarVsMassPosProxyEta", "<#it{R}>_{LeadJet} vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} Vs Mass, #eta_{Proxy}>0;#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiAntiLambdaPhiProtonStarVsMassPosProxyEta", "<#it{R}>_{LeadP} vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} Vs Mass, #eta_{Proxy}>0;#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiAntiLambdaPhiProtonStarVsMassPosProxyEta", "<#it{R}>_{SubJet} vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} Vs Mass, #eta_{Proxy}>0;#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiAntiLambdaPhiProtonStarVsMassNegProxyEta", "<#it{R}>_{LeadJet} vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} Vs Mass, #eta_{Proxy}<0;#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiAntiLambdaPhiProtonStarVsMassNegProxyEta", "<#it{R}>_{LeadP} vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} Vs Mass, #eta_{Proxy}<0;#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiAntiLambdaPhiProtonStarVsMassNegProxyEta", "<#it{R}>_{SubJet} vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} Vs Mass, #eta_{Proxy}<0;#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiAntiLambdaPhiProtonStarVsProxyEtaSign", "<#it{R}>_{LeadJet} vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} Vs #eta_{Proxy};#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*}; #eta_{Proxy} sign;<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, {2, -1, 1}});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiAntiLambdaPhiProtonStarVsProxyEtaSign", "<#it{R}>_{LeadP} vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} Vs #eta_{Proxy};#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*}; #eta_{Proxy} sign;<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, {2, -1, 1}});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiAntiLambdaPhiProtonStarVsProxyEtaSign", "<#it{R}>_{SubJet} vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} Vs #eta_{Proxy};#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*}; #eta_{Proxy} sign;<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, {2, -1, 1}});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiAntiLambdaPhiProtonStarVsProxyEta", "<#it{R}>_{LeadJet} vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} Vs #eta_{Proxy};#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*}; #eta_{Proxy};<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEta});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiAntiLambdaPhiProtonStarVsProxyEta", "<#it{R}>_{LeadP} vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} Vs #eta_{Proxy};#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*}; #eta_{Proxy};<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEta});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiAntiLambdaPhiProtonStarVsProxyEta", "<#it{R}>_{SubJet} vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} Vs #eta_{Proxy};#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*}; #eta_{Proxy};<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEta});
        // A 3D split that allows for a proper signal extraction after separating the main drivers of fake polarization signal:
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p3dRingObservableLeadJetVsPhiAntiLambdaPhiProtonStarVsProxyEtaVsMass", "<#it{R}>_{LeadJet} vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} Vs #eta_{Proxy};#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*}; #eta_{Proxy}; m_{p#pi};<#it{R}>", kTProfile3D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p3dRingObservableLeadPVsPhiAntiLambdaPhiProtonStarVsProxyEtaVsMass", "<#it{R}>_{LeadP} vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} Vs #eta_{Proxy};#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*}; #eta_{Proxy}; m_{p#pi};<#it{R}>", kTProfile3D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p3dRingObservable2ndJetVsPhiAntiLambdaPhiProtonStarVsProxyEtaVsMass", "<#it{R}>_{SubJet} vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} Vs #eta_{Proxy};#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*}; #eta_{Proxy}; m_{p#pi};<#it{R}>", kTProfile3D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
      }

      // The same nine profiles collapsed onto three mass slices:
      // (high statistics for a simpler lookup)
      if (analyseLambda) {
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/ThreeBinMass/p2dRingObservableLeadJetVsPhiLambdaPhiProtonStarVs3BinMass", "<#it{R}>_{LeadJet} vs #phi_{#Lambda}-#phi_{p}^{*} Vs mass slice;#phi_{#Lambda}-#phi_{p}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassThreeBin});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/ThreeBinMass/p2dRingObservableLeadPVsPhiLambdaPhiProtonStarVs3BinMass", "<#it{R}>_{LeadP} vs #phi_{#Lambda}-#phi_{p}^{*} Vs mass slice;#phi_{#Lambda}-#phi_{p}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassThreeBin});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/ThreeBinMass/p2dRingObservable2ndJetVsPhiLambdaPhiProtonStarVs3BinMass", "<#it{R}>_{SubJet} vs #phi_{#Lambda}-#phi_{p}^{*} Vs mass slice;#phi_{#Lambda}-#phi_{p}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassThreeBin});
      }
      if (analyseAntiLambda) {
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/ThreeBinMass/p2dRingObservableLeadJetVsPhiAntiLambdaPhiProtonStarVs3BinMass", "<#it{R}>_{LeadJet} vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} Vs mass slice;#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};m_{#bar{p}#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassThreeBin});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/ThreeBinMass/p2dRingObservableLeadPVsPhiAntiLambdaPhiProtonStarVs3BinMass", "<#it{R}>_{LeadP} vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} Vs mass slice;#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};m_{#bar{p}#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassThreeBin});
        histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/ThreeBinMass/p2dRingObservable2ndJetVsPhiAntiLambdaPhiProtonStarVs3BinMass", "<#it{R}>_{SubJet} vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} Vs mass slice;#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};m_{#bar{p}#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassThreeBin});
      }
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/ThreeBinMass/p2dRingObservableLeadJetVsPhiLambdaLikePhiProtonStarVs3BinMass", "<#it{R}>_{LeadJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs mass slice;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassThreeBin});
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/ThreeBinMass/p2dRingObservableLeadPVsPhiLambdaLikePhiProtonStarVs3BinMass", "<#it{R}>_{LeadP} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs mass slice;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassThreeBin});
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/ThreeBinMass/p2dRingObservable2ndJetVsPhiLambdaLikePhiProtonStarVs3BinMass", "<#it{R}>_{SubJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs mass slice;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassThreeBin});

      // Picking a representative split in eta and PhiAEE just to see if we manage to stabilize the background (based off p2dRingObservableEtaLambdaVsMass, but booked outside the addRingObservableFamily lambda):
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/BkgStabilizationAttempt/p2dRingObservableEtaLambdaVsMassEtaProxyPosPhiAEEPos", "<#it{R}> vs #eta_{#Lambda} vs Mass, #eta_{Proxy}>0, #phi_{AEE}>0;#eta_{#Lambda};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/BkgStabilizationAttempt/p2dRingObservableEtaLambdaVsMassEtaProxyNegPhiAEEPos", "<#it{R}> vs #eta_{#Lambda} vs Mass, #eta_{Proxy}<0, #phi_{AEE}>0;#eta_{#Lambda};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/BkgStabilizationAttempt/p2dRingObservableEtaLambdaVsMassEtaProxyPosPhiAEEPosDeltaPhiJetPos", "<#it{R}> vs #eta_{#Lambda} vs Mass, #eta_{Proxy}>0, #phi_{AEE}>0, #Delta#phi_{Jet}>0;#eta_{#Lambda};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/BkgStabilizationAttempt/p2dRingObservableEtaLambdaVsMassEtaProxyPosPhiAEEPosDeltaPhiJetNeg", "<#it{R}> vs #eta_{#Lambda} vs Mass, #eta_{Proxy}<0, #phi_{AEE}>0, #Delta#phi_{Jet}<0;#eta_{#Lambda};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisLambdaMassSigExtract});
      // Splitting in the jet coordinate, but leaving PhiAEE differential:
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/BkgStabilizationAttempt/p2dRingObservablePhiLambdaPhiProtonStarVsMassEtaProxyPosPhiEtaLambdaPos", "<#it{R}> vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} vs Mass, #eta_{Proxy}>0, #eta_{#Lambda}>0;#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/BkgStabilizationAttempt/p2dRingObservablePhiLambdaPhiProtonStarVsMassEtaProxyNegPhiEtaLambdaPos", "<#it{R}> vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} vs Mass, #eta_{Proxy}<0, #eta_{#Lambda}>0;#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/BkgStabilizationAttempt/p2dRingObservablePhiLambdaPhiProtonStarVsMassEtaProxyPosPhiEtaLambdaPosDeltaPhiJetPos", "<#it{R}> vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} vs Mass, #eta_{Proxy}>0, #eta_{#Lambda}>0, #Delta#phi_{Jet}>0;#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/BkgStabilizationAttempt/p2dRingObservablePhiLambdaPhiProtonStarVsMassEtaProxyPosPhiEtaLambdaPosDeltaPhiJetNeg", "<#it{R}> vs #phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*} vs Mass, #eta_{Proxy}<0, #eta_{#Lambda}>0, #Delta#phi_{Jet}<0;#phi_{#bar{#Lambda}}-#phi_{#bar{p}}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
    } // end doFakePolDiagnosticsQA bookings

    // Integrated observable for events with NLambda+NAntiLambda V0s per event
    // (an interesting measurement of correlation between <R> and Lambda-like V0s multiplicity. A proxy of covariance)
    // (calculated for leading jets only)
    histos.add("IntegratedCuts/pRingVsNV0s", "pRingVsNV0s; N_{#Lambda}+N_{#bar{#Lambda}};<#it{R}>", kTProfile, {{20, 0, 20}}); // See hNV0sVsCentrality below for the correlation between number of V0s and centrality
    histos.add("hNV0sVsCentrality", "hNV0sVsCentrality; N_{#Lambda}+N_{#bar{#Lambda}};Centrality (%)", kTH2D, {{20, 0, 20}, axisConfigurations.axisCentrality});

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
    histos.add("KinematicsQA/Jet/hLeadJetEta", "hLeadJetEta;#eta;Counts", kTH1D, {axisConfigurations.axisEta});
    histos.add("KinematicsQA/Jet/hSubLeadJetEta", "hSubLeadJetEta;#eta;Counts", kTH1D, {axisConfigurations.axisEta});
    histos.add("KinematicsQA/Jet/hLeadPEta", "hLeadPEta;#eta;Counts", kTH1D, {axisConfigurations.axisEta});

    // Proxy Phi QA:
    histos.add("KinematicsQA/Jet/hLeadJetPhi", "hLeadJetPhi;#varphi;Counts", kTH1D, {axisConfigurations.axisPhi});
    histos.add("KinematicsQA/Jet/hSubLeadJetPhi", "hSubLeadJetPhi;#varphi;Counts", kTH1D, {axisConfigurations.axisPhi});
    histos.add("KinematicsQA/Jet/hLeadPPhi", "hLeadPPhi;#varphi;Counts", kTH1D, {axisConfigurations.axisPhi});

    // Counting the number of jets/proxies themselves (these count at most once per event) -- Similar to the FOLDER/QA/hPtJet counters:
    histos.add("KinematicsQA/Jet/hJetCounterPtJet", "hJetCounterPtJet; p_{T}^{Jet} (GeV/c)", kTH1D, {axisConfigurations.axisJetPt});
    histos.add("KinematicsQA/Jet/hJetCounterPtLeadP", "hJetCounterPtLeadP; p_{T}^{LeadP} (GeV/c)", kTH1D, {axisConfigurations.axisJetPt});
    histos.add("KinematicsQA/Jet/hJetCounterPt2ndJet", "hJetCounterPt2ndJet; p_{T}^{SubJet} (GeV/c)", kTH1D, {axisConfigurations.axisJetPt});

    // Proxy Eta vs Proxy Phi:
    histos.add("KinematicsQA/Jet/h2dLeadJetEtaVsPhi", "Lead Jet #eta Vs #phi;#eta;#phi_{Proxy} [rad];Counts", kTH2D, {axisConfigurations.axisEta, axisConfigurations.axisPhi});
    histos.add("KinematicsQA/Jet/h2dSubLeadJetEtaVsPhi", "SubLead Jet #eta Vs #phi;#eta;#phi_{Proxy} [rad];Counts", kTH2D, {axisConfigurations.axisEta, axisConfigurations.axisPhi});
    histos.add("KinematicsQA/Jet/h2dLeadPEtaVsPhi", "Lead Ptc #eta Vs #phi;#eta;#phi_{Proxy} [rad];Counts", kTH2D, {axisConfigurations.axisEta, axisConfigurations.axisPhi});


    // Proxy Eta vs PVz:
    histos.add("KinematicsQA/Jet/h2dLeadJetEtaVsPVz", "Lead Jet #eta Vs PVz;#eta;Primary Vertex Z [cm];Counts", kTH2D, {axisConfigurations.axisEta, axisConfigurations.axisPVz});
    histos.add("KinematicsQA/Jet/h2dSubLeadJetEtaVsPVz", "SubLead Jet #eta Vs PVz;#eta;Primary Vertex Z [cm];Counts", kTH2D, {axisConfigurations.axisEta, axisConfigurations.axisPVz});
    histos.add("KinematicsQA/Jet/h2dLeadPEtaVsPVz", "Lead Ptc #eta Vs PVz;#eta;Primary Vertex Z [cm];Counts", kTH2D, {axisConfigurations.axisEta, axisConfigurations.axisPVz});

    // For building and event-mixing-like procedure similar to forceDatalikeJet:
    // if (doJetProxy5dQA) {
    //   histos.add("KinematicsQA/Jet/h5dLeadJetEtaPhiPtPVzCent", "h5dLeadJetEtaPhiPtPVzCent;#eta;#phi;p_{t};Primary Vertex Z [cm];Centrality (%);Counts", kTHnSparseF,
    //              {axisConfigurations.axisEtaCoarse, axisConfigurations.axisPhi, axisConfigurations.axisJetPt, axisConfigurations.axisPVz, axisConfigurations.axisCentrality});
    //   histos.add("KinematicsQA/Jet/h5dSubLeadJetEtaPhiPtPVzCent", "h5dSubLeadJetEtaPhiPtPVzCent;#eta;#phi;p_{t};Primary Vertex Z [cm];Centrality (%);Counts", kTHnSparseF,
    //              {axisConfigurations.axisEtaCoarse, axisConfigurations.axisPhi, axisConfigurations.axisJetPt, axisConfigurations.axisPVz, axisConfigurations.axisCentrality});
    //   histos.add("KinematicsQA/Jet/h5dLeadPEtaPhiPtPVzCent", "h5dLeadPEtaPhiPtPVzCent;#eta;#phi;p_{t};Primary Vertex Z [cm];Centrality (%);Counts", kTHnSparseF,
    //              {axisConfigurations.axisEtaCoarse, axisConfigurations.axisPhi, axisConfigurations.axisJetPt, axisConfigurations.axisPVz, axisConfigurations.axisCentrality});
    // }

    // doMixedEventProxies QA: gauge the size of the "too few collisions per bin" problem (see resonanceMergeDF.cxx)
    // Booked per proxy, since the three mixings are independent and can succeed/fail at different rates.
    if (fakePolSwitches.doMixedEventProxies && qaSwitches.doEventMixingQA) {
      // These histograms are filled in the collision loop, so we are QAing only the collisions with V0s
      histos.add("EventMixingQA/CollLoopOutcome/hMixedEventLeadPOutcome", "hMixedEventLeadPOutcome;Outcome (0=skipped, 1=found);Counts", kTH1D, {{2, -0.5f, 1.5f}});
      histos.add("EventMixingQA/CollLoopOutcome/hMixedEventLeadJetOutcome", "hMixedEventLeadJetOutcome;Outcome (0=skipped, 1=found);Counts", kTH1D, {{2, -0.5f, 1.5f}});
      histos.add("EventMixingQA/CollLoopOutcome/hMixedEventSubJetOutcome", "hMixedEventSubJetOutcome;Outcome (0=skipped, 1=found);Counts", kTH1D, {{2, -0.5f, 1.5f}});
        // 2D histograms on every mixing variable:
      histos.add("EventMixingQA/CollLoopOutcome/h2dMixedEventLeadPOutcomeVsTargetZVtx", "h2dMixedEventLeadPOutcomeVsTargetZVtx;Outcome (0=skipped, 1=found);Target primary Vertex Z [cm];Counts", kTH2D, {{2, -0.5f, 1.5f}, axisConfigurations.axisPVz});
      histos.add("EventMixingQA/CollLoopOutcome/h2dMixedEventLeadJetOutcomeVsTargetZVtx", "h2dMixedEventLeadJetOutcomeVsTargetZVtx;Outcome (0=skipped, 1=found);Target primary Vertex Z [cm];Counts", kTH2D, {{2, -0.5f, 1.5f}, axisConfigurations.axisPVz});
      histos.add("EventMixingQA/CollLoopOutcome/h2dMixedEventSubJetOutcomeVsTargetZVtx", "h2dMixedEventSubJetOutcomeVsTargetZVtx;Outcome (0=skipped, 1=found);Target primary Vertex Z [cm];Counts", kTH2D, {{2, -0.5f, 1.5f}, axisConfigurations.axisPVz});

      histos.add("EventMixingQA/CollLoopOutcome/h2dMixedEventLeadPOutcomeVsTargetCentrality", "h2dMixedEventLeadPOutcomeVsTargetCentrality;Outcome (0=skipped, 1=found);Target centrality (%);Counts", kTH2D, {{2, -0.5f, 1.5f}, axisConfigurations.axisCentrality});
      histos.add("EventMixingQA/CollLoopOutcome/h2dMixedEventLeadJetOutcomeVsTargetCentrality", "h2dMixedEventLeadJetOutcomeVsTargetCentrality;Outcome (0=skipped, 1=found);Target centrality (%);Counts", kTH2D, {{2, -0.5f, 1.5f}, axisConfigurations.axisCentrality});
      histos.add("EventMixingQA/CollLoopOutcome/h2dMixedEventSubJetOutcomeVsTargetCentrality", "h2dMixedEventSubJetOutcomeVsTargetCentrality;Outcome (0=skipped, 1=found);Target centrality (%);Counts", kTH2D, {{2, -0.5f, 1.5f}, axisConfigurations.axisCentrality});

      histos.add("EventMixingQA/CollLoopOutcome/h2dMixedEventLeadPOutcomeVsTargetProxyPt", "h2dMixedEventLeadPOutcomeVsTargetProxyPt;Outcome (0=skipped, 1=found);Target proxy p_{t} [GeV/c];Counts", kTH2D, {{2, -0.5f, 1.5f}, axisConfigurations.axisJetPt});
      histos.add("EventMixingQA/CollLoopOutcome/h2dMixedEventLeadJetOutcomeVsTargetProxyPt", "h2dMixedEventLeadJetOutcomeVsTargetProxyPt;Outcome (0=skipped, 1=found);Target proxy p_{t} [GeV/c];Counts", kTH2D, {{2, -0.5f, 1.5f}, axisConfigurations.axisJetPt});
      histos.add("EventMixingQA/CollLoopOutcome/h2dMixedEventSubJetOutcomeVsTargetProxyPt", "h2dMixedEventSubJetOutcomeVsTargetProxyPt;Outcome (0=skipped, 1=found);Target proxy p_{t} [GeV/c];Counts", kTH2D, {{2, -0.5f, 1.5f}, axisConfigurations.axisJetPt});

      // QA at LUT building time:
      histos.add("EventMixingQA/hMixedEventLeadPWindowNeighbours", "hMixedEventLeadPWindowNeighbours;Neighbours found in bin window;Counts", kTH1D, {axisConfigurations.axisMixCandidates});
      histos.add("EventMixingQA/hMixedEventLeadJetWindowNeighbours", "hMixedEventLeadJetWindowNeighbours;Neighbours found in bin window;Counts", kTH1D, {axisConfigurations.axisMixCandidates});
      histos.add("EventMixingQA/hMixedEventSubJetWindowNeighbours", "hMixedEventSubJetWindowNeighbours;Neighbours found in bin window;Counts", kTH1D, {axisConfigurations.axisMixCandidates});

      histos.add("EventMixingQA/AnglrCorrltns/h2dMixedLeadPEtaVsLeadPEta", "MixedLeadP #eta vs LeadP #eta;MixedLeadP #eta; LeadP #eta;Counts", kTH2D, {axisConfigurations.axisEtaCoarse, axisConfigurations.axisEtaCoarse});
      histos.add("EventMixingQA/AnglrCorrltns/h2dMixedLeadJetEtaVsLeadJetEta", "MixedLeadJet #eta vs LeadJet #eta;MixedLeadJet #eta; LeadJet #eta;Counts", kTH2D, {axisConfigurations.axisEtaCoarse, axisConfigurations.axisEtaCoarse});
      histos.add("EventMixingQA/AnglrCorrltns/h2dMixedSubJetEtaVsSubJetEta", "MixedSubJet #eta vs SubJet #eta;MixedSubJet #eta; SubJet #eta;Counts", kTH2D, {axisConfigurations.axisEtaCoarse, axisConfigurations.axisEtaCoarse});
      histos.add("EventMixingQA/AnglrCorrltns/h2dMixedLeadPPhiVsLeadPPhi", "MixedLeadP #phi vs LeadP #phi;MixedLeadP #phi; LeadP #phi;Counts", kTH2D, {axisConfigurations.axisPhi, axisConfigurations.axisPhi});
      histos.add("EventMixingQA/AnglrCorrltns/h2dMixedLeadJetPhiVsLeadJetPhi", "MixedLeadJet #phi vs LeadJet #phi;MixedLeadJet #phi; LeadJet #phi;Counts", kTH2D, {axisConfigurations.axisPhi, axisConfigurations.axisPhi});
      histos.add("EventMixingQA/AnglrCorrltns/h2dMixedSubJetPhiVsSubJetPhi", "MixedSubJet #phi vs SubJet #phi;MixedSubJet #phi; SubJet #phi;Counts", kTH2D, {axisConfigurations.axisPhi, axisConfigurations.axisPhi});

      // Collision-index proximity QA:
      // (To understand possible continous readout effects on the choice of event mixing sources -- do notice index and time are both monotonic, but there is a conversion factor)
      // The shape of the whole candidate pool:
      histos.add("EventMixingQA/IndexQA/hMixedEventLeadPDeltaIndexEligible", "hMixedEventLeadPDeltaIndexEligible;#Delta collision index (pair);Counts", kTH1D, {axisConfigurations.axisDeltaCollisionIndex});
      histos.add("EventMixingQA/IndexQA/hMixedEventLeadJetDeltaIndexEligible", "hMixedEventLeadJetDeltaIndexEligible;#Delta collision index (pair);Counts", kTH1D, {axisConfigurations.axisDeltaCollisionIndex});
      histos.add("EventMixingQA/IndexQA/hMixedEventSubJetDeltaIndexEligible", "hMixedEventSubJetDeltaIndexEligible;#Delta collision index (pair);Counts", kTH1D, {axisConfigurations.axisDeltaCollisionIndex});

      // What the reservoir actually picked:
      // (should be equally as narrow as the whole candidate pool, if no preferential mixing exists)
      // This number may be higher than IndexEligible: Eligible has one entry per accepted pair, while Selected has one entry per distinct collision that has at least one mixing candidate.
      histos.add("EventMixingQA/IndexQA/hMixedEventLeadPDeltaIndexSelected", "hMixedEventLeadPDeltaIndexSelected;#Delta collision index (selected partner);Counts", kTH1D, {axisConfigurations.axisDeltaCollisionIndexNonAbs});
      histos.add("EventMixingQA/IndexQA/hMixedEventLeadJetDeltaIndexSelected", "hMixedEventLeadJetDeltaIndexSelected;#Delta collision index (selected partner);Counts", kTH1D, {axisConfigurations.axisDeltaCollisionIndexNonAbs});
      histos.add("EventMixingQA/IndexQA/hMixedEventSubJetDeltaIndexSelected", "hMixedEventSubJetDeltaIndexSelected;#Delta collision index (selected partner);Counts", kTH1D, {axisConfigurations.axisDeltaCollisionIndexNonAbs});

      // Event mixing source QA -- How repeated is each collision in the mix:
      // For each source collision for mixing, counts how many times it was used. Flat distribution is ideal.
      histos.add("EventMixingQA/SourceUsage/hMixedEventLeadPSourceUsageCount", "hMixedEventLeadPSourceUsageCount;Times collision was used;Counts", kTH1D, {{50, -0.5f, 49.5f}});
      histos.add("EventMixingQA/SourceUsage/hMixedEventLeadJetSourceUsageCount", "hMixedEventLeadJetSourceUsageCount;Times collision was used;Counts", kTH1D, {{50, -0.5f, 49.5f}});
      histos.add("EventMixingQA/SourceUsage/hMixedEventSubJetSourceUsageCount", "hMixedEventSubJetSourceUsageCount;Times collision was used;Counts", kTH1D, {{50, -0.5f, 49.5f}});
      // Useful TProfiles -- the mean number of times a given source was used, as a function of eta or phi
      // (more convenient than a single TH1, as it gives an average, not the raw counter)
      histos.add("EventMixingQA/SourceUsage/pMixedEventLeadPSourceUsageVsEta", "pMixedEventLeadPSourceUsageVsEta;Source #eta;<Times used>", kTProfile, {axisConfigurations.axisEtaCoarse});
      histos.add("EventMixingQA/SourceUsage/pMixedEventLeadPSourceUsageVsPhi", "pMixedEventLeadPSourceUsageVsPhi;Source #varphi;<Times used>", kTProfile, {axisConfigurations.axisPhi});
      histos.add("EventMixingQA/SourceUsage/pMixedEventLeadJetSourceUsageVsEta", "pMixedEventLeadJetSourceUsageVsEta;Source #eta;<Times used>", kTProfile, {axisConfigurations.axisEtaCoarse});
      histos.add("EventMixingQA/SourceUsage/pMixedEventLeadJetSourceUsageVsPhi", "pMixedEventLeadJetSourceUsageVsPhi;Source #varphi;<Times used>", kTProfile, {axisConfigurations.axisPhi});
      histos.add("EventMixingQA/SourceUsage/pMixedEventSubJetSourceUsageVsEta", "pMixedEventSubJetSourceUsageVsEta;Source #eta;<Times used>", kTProfile, {axisConfigurations.axisEtaCoarse});
      histos.add("EventMixingQA/SourceUsage/pMixedEventSubJetSourceUsageVsPhi", "pMixedEventSubJetSourceUsageVsPhi;Source #varphi;<Times used>", kTProfile, {axisConfigurations.axisPhi});

      // Source-vs-target kinematics for the borrowed proxy, filled before applyProxyDistortion runs:
      // These include "V0less" collisions that can be mixing sources, so it probes a different subset of events from hMixedEventLeadPOutcome's.
      histos.add("EventMixingQA/SourceTargetDeltas/hMixedEventLeadPDeltaPt", "hMixedEventLeadPDeltaPt;#Delta p_{T};Counts", kTH1D, {axisConfigurations.axisMixDeltaPt});
      histos.add("EventMixingQA/SourceTargetDeltas/hMixedEventLeadJetDeltaPt", "hMixedEventLeadJetDeltaPt;#Delta p_{T};Counts", kTH1D, {axisConfigurations.axisMixDeltaPt});
      histos.add("EventMixingQA/SourceTargetDeltas/hMixedEventSubJetDeltaPt", "hMixedEventSubJetDeltaPt;#Delta p_{T};Counts", kTH1D, {axisConfigurations.axisMixDeltaPt});

      histos.add("EventMixingQA/SourceTargetDeltas/hMixedEventLeadPDeltaZvtx", "hMixedEventLeadPDeltaZvtx;#Delta Z_{Vtx};Counts", kTH1D, {axisConfigurations.axisMixDeltaZvtx});
      histos.add("EventMixingQA/SourceTargetDeltas/hMixedEventLeadJetDeltaZvtx", "hMixedEventLeadJetDeltaZvtx;#Delta Z_{Vtx};Counts", kTH1D, {axisConfigurations.axisMixDeltaZvtx}); // There should be no noticeable change between LeadP and LeadJet's collision-related histograms (if they don't bias event selection)
      histos.add("EventMixingQA/SourceTargetDeltas/hMixedEventSubJetDeltaZvtx", "hMixedEventSubJetDeltaZvtx;#Delta Z_{Vtx};Counts", kTH1D, {axisConfigurations.axisMixDeltaZvtx});

      histos.add("EventMixingQA/SourceTargetDeltas/hMixedEventLeadPDeltaCent", "hMixedEventLeadPDeltaCent;Centrality(%);Counts", kTH1D, {axisConfigurations.axisMixDeltaCentrality});
      histos.add("EventMixingQA/SourceTargetDeltas/hMixedEventLeadJetDeltaCent", "hMixedEventLeadJetDeltaCent;Centrality(%);Counts", kTH1D, {axisConfigurations.axisMixDeltaCentrality});
      histos.add("EventMixingQA/SourceTargetDeltas/hMixedEventSubJetDeltaCent", "hMixedEventSubJetDeltaCent;Centrality(%);Counts", kTH1D, {axisConfigurations.axisMixDeltaCentrality});
      
      histos.add("EventMixingQA/SourceTargetDeltas/hMixedEventLeadPDeltaEta", "hMixedEventLeadPDeltaEta;#Delta#eta;Counts", kTH1D, {axisConfigurations.axisMixDeltaEta});
      histos.add("EventMixingQA/SourceTargetDeltas/hMixedEventLeadJetDeltaEta", "hMixedEventLeadJetDeltaEta;#Delta#eta;Counts", kTH1D, {axisConfigurations.axisMixDeltaEta});
      histos.add("EventMixingQA/SourceTargetDeltas/hMixedEventSubJetDeltaEta", "hMixedEventSubJetDeltaEta;#Delta#eta;Counts", kTH1D, {axisConfigurations.axisMixDeltaEta});
      
      histos.add("EventMixingQA/SourceTargetDeltas/hMixedEventLeadPDeltaPhi", "hMixedEventLeadPDeltaPhi;#Delta#varphi;Counts", kTH1D, {axisConfigurations.axisMixDeltaPhi});
      histos.add("EventMixingQA/SourceTargetDeltas/hMixedEventLeadJetDeltaPhi", "hMixedEventLeadJetDeltaPhi;#Delta#varphi;Counts", kTH1D, {axisConfigurations.axisMixDeltaPhi});
      histos.add("EventMixingQA/SourceTargetDeltas/hMixedEventSubJetDeltaPhi", "hMixedEventSubJetDeltaPhi;#Delta#varphi;Counts", kTH1D, {axisConfigurations.axisMixDeltaPhi});

      // "Closure test" for the binning policy:
      // Every entry's delta must be <= than the width of its own y bin (e.g., a 2-4 GeV/c target cannot borrow from more than 2 GeV/c away).
      // (LeadP only as it would verify the same thing)
      histos.add("EventMixingQA/SourceTargetDeltas/h2dMixedEventLeadPDeltaPtVsTargetPt", "h2dMixedEventLeadPDeltaPtVsTargetPt;#Delta p_{T} (GeV/c);Target proxy p_{T} (GeV/c);Counts", kTH2D, {axisConfigurations.axisMixDeltaPt, axisConfigurations.axisJetPt});
      histos.add("EventMixingQA/SourceTargetDeltas/h2dMixedEventLeadPDeltaZvtxVsTargetZvtx", "h2dMixedEventLeadPDeltaZvtxVsTargetZvtx;#Delta Z_{Vtx} (cm);Target Z_{Vtx} (cm);Counts", kTH2D, {axisConfigurations.axisMixDeltaZvtx, axisConfigurations.axisPVz});
      histos.add("EventMixingQA/SourceTargetDeltas/h2dMixedEventLeadPDeltaCentVsTargetCent", "h2dMixedEventLeadPDeltaCentVsTargetCent;#Delta Centrality (%);Target centrality (%);Counts", kTH2D, {axisConfigurations.axisMixDeltaCentrality, axisConfigurations.axisCentrality});

      // Index Vs "Delta mixed variable" TH2s to check if the width changes with indexing (possible evtSplittings or cannibalizations):
      histos.add("EventMixingQA/IndexQA/h2dMixedEventLeadPDeltaPtVsDeltaIndex", "h2dMixedEventLeadPDeltaPtVsDeltaIndex;#Delta collision index;#Delta p_{T} (GeV/c);Counts", kTH2D, {axisConfigurations.axisMixDeltaIndexCoarse, axisConfigurations.axisMixDeltaPtCoarse});
      histos.add("EventMixingQA/IndexQA/h2dMixedEventLeadPDeltaZvtxVsDeltaIndex", "h2dMixedEventLeadPDeltaZvtxVsDeltaIndex;#Delta collision index;#Delta Z_{Vtx} (cm);Counts", kTH2D, {axisConfigurations.axisMixDeltaIndexCoarse, axisConfigurations.axisMixDeltaZvtxCoarse});
      histos.add("EventMixingQA/IndexQA/h2dMixedEventLeadPDeltaCentVsDeltaIndex", "h2dMixedEventLeadPDeltaCentVsDeltaIndex;#Delta collision index;#Delta Centrality (%);Counts", kTH2D, {axisConfigurations.axisMixDeltaIndexCoarse, axisConfigurations.axisMixDeltaCentCoarse});

      // Counting bit-exact coincidences of values (QA for event mixing to make sure no auto-correlated mixing happens):
      histos.add("EventMixingQA/IdentityChecks/hMixedEventLeadPIdentityFlags", "hMixedEventLeadPIdentityFlags;0: pt, 1: eta, 2: phi, 3: Zvtx;Exact matches", kTH1D, {{4, -0.5f, 3.5f}});
      // Angular coincidence check on a log scale (alternative):
      histos.add("EventMixingQA/IdentityChecks/hMixedEventLeadPLogAngularSep", "hMixedEventLeadPLogAngularSep;log_{10}(#sqrt{#Delta#eta^{2} + #Delta#varphi^{2}});Counts", kTH1D, {{150, -7.f, 0.5f}});
      // Bit-identical Zvtx value check at the LUT building time (characterization of "Birthday paradox" effect):
      histos.add("EventMixingQA/IdentityChecks/hDuplicateZvtxCollisions", "hDuplicateZvtxCollisions;Collisions sharing one Zvtx value;Counts", kTH1D, {{101, 1.5f, 102.5f}});

      // How many mixing candidates a collision actually had:
      histos.add("EventMixingQA/CollLoopOutcome/hMixedEventLeadPCandidates", "hMixedEventLeadPCandidates;Candidates seen by this collision;Counts", kTH1D, {axisConfigurations.axisMixCandidates});
      histos.add("EventMixingQA/CollLoopOutcome/hMixedEventLeadJetCandidates", "hMixedEventLeadJetCandidates;Candidates seen by this collision;Counts", kTH1D, {axisConfigurations.axisMixCandidates});
      histos.add("EventMixingQA/CollLoopOutcome/hMixedEventSubJetCandidates", "hMixedEventSubJetCandidates;Candidates seen by this collision;Counts", kTH1D, {axisConfigurations.axisMixCandidates});

      // The same count resolved against the pt bin the collision was mixed in (source and target share it):
      histos.add("EventMixingQA/CollLoopOutcome/h2dMixedLeadPCandidatesVsPt", "h2dMixedLeadPCandidatesVsPt;Candidates;LeadP p_{T} (GeV/c);Counts", kTH2D, {axisConfigurations.axisMixCandidates, axisConfigurations.axisJetPt});
      histos.add("EventMixingQA/CollLoopOutcome/h2dMixedLeadJetCandidatesVsPt", "h2dMixedLeadJetCandidatesVsPt;Candidates;LeadJet p_{T} (GeV/c);Counts", kTH2D, {axisConfigurations.axisMixCandidates, axisConfigurations.axisJetPt});
      histos.add("EventMixingQA/CollLoopOutcome/h2dMixedSubJetCandidatesVsPt", "h2dMixedSubJetCandidatesVsPt;Candidates;SubJet p_{T} (GeV/c);Counts", kTH2D, {axisConfigurations.axisMixCandidates, axisConfigurations.axisJetPt});
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

  /// \brief One jet proxy (leadP, leadJet or subJet). Bound by reference so applyProxyDistortion() edits in-place.
  struct ProxyState {
    bool& hasValidProxy; //! whether the proxy is valid. Re-evaluated against minPtThreshold after distortion
    float& pt;
    float& eta;
    float& phi;
    XYZVector& unitVec; //! the proxy direction as a unit vector
  };

  /// \brief The cache slots holding the source proxy for forcePreviousJet and doMixedEventProxies. Read-only and by reference.
  /// \note  Fallback skips this event using ProxyState::hasValidProxy.
  /// \note  forcePreviousJet fills these in the main loop, from the previous analysed collision's own proxy.
  //         doMixedEventProxies instead overwrites the caller's cache fields with this collision's mixed proxy right before the call.
  struct ProxyCacheRef {
    bool& hadProxy;
    float& pt; //! the borrowed proxy carries its own pt: it is adopted, not recomputed
    float& eta;
    float& phi;
  };

  /// \brief Applies whichever fakePolSwitches distortion is active to a jet-proxy direction, in place. No-op if none are on.
  /// \param proxy input/edited in-place: the proxy's kinematics, overwritten by whichever distortion is active.
  /// \param minPtThreshold pT threshold for re-evaluating hasValidProxy after distortion.
  /// \param cache input only: the previous-jet/mixed-event source proxy. The caller keeps it up to date. not this function.
  /// \param etaDist,phiDist sampling distributions for forceDatalikeJet (drawn from the task's own rng member).
  /// \note  Shared across leadP/leadJet/subJet: the caller resolves which proxy-specific procedure (e.g., LUT for evtMixing) applies.
  // Helper to modify the jet direction for QA and for spurious signal baseline removal tests:
  inline void applyProxyDistortion(ProxyState proxy, float minPtThreshold, float maxAbsEta, ProxyCacheRef cache,
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
      if (std::abs(proxy.unitVec.X()) > 0.99) {
        perpVec = XYZVector(-proxy.unitVec.Z(), 0., proxy.unitVec.X()).Unit(); // Cross product with Y-axis (0, 1, 0)
      } else {
        perpVec = XYZVector(0., proxy.unitVec.Z(), -proxy.unitVec.Y()).Unit(); // Cross product with X-axis (1, 0, 0)
      }

      // Now we rotate around the jet axis by a random angle, just to make sure we are not introducing a bias in the QA:
      // We will use Rodrigues' rotation formula (v_rot = v*cos(randomAngle) + (Jet \cross v)*sin(randomAngle))
      const double randomAngle = randomGen.Uniform(0., constants::math::TwoPI);
      proxy.unitVec = perpVec * std::cos(randomAngle) + proxy.unitVec.Cross(perpVec) * std::sin(randomAngle);
    } else if (fakePolSwitches.forceJetDirectionSmudge) {
      // Smear the jet direction by a small random angle to estimate sensitivity to
      // jet axis uncertainty. We rotate the jet axis by angle theta around a uniformly
      // random perpendicular axis -- this is isotropic and coordinate-independent,
      // unlike smearing eta and phi separately (which would break azimuthal symmetry
      // around the jet axis and depend on where in eta the jet sits).

      // 1) We pick a uniformly random axis perpendicular to the jet.
      // (re-using the same Rodrigues formula as in the forcePerpToJet block above)
      XYZVector perpVec;
      if (std::abs(proxy.unitVec.X()) > 0.99) {
        perpVec = XYZVector(-proxy.unitVec.Z(), 0., proxy.unitVec.X()).Unit(); // Cross product with Y-axis (0, 1, 0)
      } else {
        perpVec = XYZVector(0., proxy.unitVec.Z(), -proxy.unitVec.Y()).Unit(); // Cross product with X-axis (1, 0, 0)
      }

      // Rotate perpVec around the jet axis by a uniform random azimuth to get
      // a uniformly distributed random perpendicular direction (the smear axis):
      const double smearAzimuth = randomGen.Uniform(0., constants::math::TwoPI);
      XYZVector smearAxis = perpVec * std::cos(smearAzimuth) + proxy.unitVec.Cross(perpVec) * std::sin(smearAzimuth);

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
      proxy.unitVec = proxy.unitVec * std::cos(smearAngle) + smearAxis.Cross(proxy.unitVec) * std::sin(smearAngle);
      // Also, rotation preserves the norm, so no re-normalisation is needed for this to be a unit vector.
    } else if (fakePolSwitches.forceRandJet) {
      // This randomization was made different for each proxy (LeadP, LeadJet, SubLeadJet): bear that in mind!
      // 1) Uniformly sample cos(theta) and phi to ensure an isotropic distribution (could also use TRandom::Sphere as well, but may be slower)
      // Notice that uniformly sampling theta would make the distribution non-isotropic, thus we use cos(theta)!
      const double cosTheta = randomGen.Uniform(-1., 1.);
      const double sinTheta = std::sqrt(1. - cosTheta * cosTheta);
      const double randPhi = randomGen.Uniform(0., constants::math::TwoPI);

      // 2) Construct the new random unit vector (there is no need to use the magnitude at all! We only need direction here):
      proxy.unitVec = XYZVector(sinTheta * std::cos(randPhi), sinTheta * std::sin(randPhi), cosTheta);
    } else if (borrowedPhysicalProxy) { // forcePreviousJet or doMixedEventProxies: a real proxy reconstructed in another collision
      if (cache.hadProxy) {
        // Adopt the source proxy wholesale:
        // From here on this collision is treated as if it had had that leading particle/jet all along.
        proxy.pt = cache.pt;
        proxy.eta = cache.eta;
        proxy.phi = cache.phi;
        const double inverseCoshEta = 1.0 / std::cosh(cache.eta);
        const double sinPhi = std::sin(cache.phi);
        const double cosPhi = std::cos(cache.phi);
        proxy.unitVec = XYZVector(cosPhi * inverseCoshEta, sinPhi * inverseCoshEta, std::tanh(cache.eta));
      } else {
        // No source proxy for this collision: the previous collision had none, or the mixing bins were too sparse.
        // Either way this collision cannot be used.
        proxy.hasValidProxy = false;
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
      proxy.unitVec = XYZVector(cosPhi * inverseCoshEta, sinPhi * inverseCoshEta, std::tanh(proxy.eta));
    }

    if (proxy.hasValidProxy) { // If you don't check this flag here, the borrowed-proxy miss above would be silently overwritten
      // if (borrowedPhysicalProxy) {
      //   // The adopted proxy is a real one that already passed this same cut when the pool was built
      //   // (minimum-pT gates on the mixing LUTs, hasValidProxy on the previous collision).
      //   // Re-checked only as a guard: it should never fire.
      //   proxy.hasValidProxy = proxy.pt > minPtThreshold;
      // }
      // Artificial proxy handling -- only the direction was "invented", so pT, phi and eta are rebuilt from proxy.unitVec.
      // (Without this, later kinematic selections and QAs are inconsistent with the calculated ring observable)
      if (!borrowedPhysicalProxy) {
        // For stability (Rho is the projection on the transverse plane, badly behaved for high |eta|):
        const double transverseNorm = std::max(proxy.unitVec.Rho(), 1e-12); // Stability guard

        // Our choice is preserving |p| across the change of direction, so pT follows the new polar angle:
        proxy.pt = beforeDistTotalMomentum * transverseNorm;

        if (!fakePolSwitches.forceDatalikeJet) { // DatalikeJet doesn't require recalculating these directions
          // Recalculate phi:
          proxy.phi = RecoDecay::constrainAngle(std::atan2(proxy.unitVec.Y(), proxy.unitVec.X()), 0.0f); // atan2 outputs [-PI, PI), and DataModel convention was [0,2PI) as per FastJet's phi() getter

          // Stable eta computation:
          // Stabler than 0.5 * std::log((1. + cosTheta) / (1. - cosTheta))
          // (for forceDatalikeJet this reproduces the eta/phi its own branch sampled, so one path covers all four modes)
          proxy.eta = std::asinh(proxy.unitVec.Z() / transverseNorm);
        }

        // These are not jets we measured, so the pT selection is optional here: it exists to let the bias
        // it introduces be measured.
        if (fakePolSwitches.gatePtOnArtificialProxies)
          proxy.hasValidProxy = proxy.pt > minPtThreshold;
        // forceRandJet draws isotropic jets in solid angle, so this puts all "artificial" modes under data-like acceptance:
        if (fakePolSwitches.gateEtaOnArtificialProxies)
          proxy.hasValidProxy = std::abs(proxy.eta) < maxAbsEta;
      }
    }
  }

  // Caching the previous collision's jet directions -- A feature for forcePreviousJet QA:
  // Slots holding a borrowed proxy (one set per proxy type).
  // One instance for prevJet carried between collisions, and one for mixedProxy in each collision.
  struct ProxyCacheSlots {
    // Leading jet
    bool hadLeadJet;
    float leadJetPt;
    float leadJetEta;
    float leadJetPhi;
    // Subleading jet
    bool hadSubJet;
    float subJetPt;
    float subJetEta;
    float subJetPhi;
    // Leading particle
    bool hadLeadP;
    float leadPPt;
    float leadPEta;
    float leadPPhi;
  };

  /// \brief Per-collision leading/subleading jet, computed once per dataframe and shared by both the main loop below
  /// (instead of re-scanning RingJets once per resampling pass) and the leadJet/subJet event mixing functions:
  struct JetProxyCache {
    bool hasValidLeadingJet = false;
    float leadingJetPt = -1.f;
    float leadingJetEta = 0.f;
    float leadingJetPhi = 0.f;
    bool hasValidSubJet = false;
    float subleadingJetPt = -1.f;
    float subleadingJetEta = 0.f;
    float subleadingJetPhi = 0.f;
  };

  /// \brief The collision-level mixing axes' cache. Cached once per collision so the pooling never needs a table iterator.
  struct MixingAxes {
    float zvtx = -999.f;
    float centrality = -1.f;
  };

  /// \brief Leading particle of a collision, resolved once in the pre-pass so the mixing never has to slice RingLeadPs.
  ///        Similar to JetProxyCache in its philosophy. Kept separate for a separate loop's accessing pattern.
  struct LeadPCache {
    bool isValid = false;
    float pt = -999.f;
    float eta = -999.f;
    float phi = -999.f;
  };

  // A simple struct for doMixedEventProxies. Each proxy (leadP/leadJet/subJet) gets its own independent cache.
  // (stores information from the borrowed event and the borrowed jets)
  struct MixedProxyInfo {
    float pt;
    float eta;
    float phi;
    float zvtx; //! Knowingly repeated for the LeadP, LeadJet and SubJet, so memory is occupied 3x for this same value.
    float centrality;
    int64_t sourceCollisionId = -1;
  };

  /// \brief Uniform random pick among each target's candidate window, via reservoir sampling
  ///        (cannibalization of proxies by neighbouring collisions in Continuous Readout is not a worry as ITS hits are being demanded)
  /// \note  Reuses the task's own rng member rather than a separate generator per proxy.
  void reservoirInsert(std::vector<int32_t>& candidateCount, std::vector<MixedProxyInfo>& lut,
                       size_t targetSlot, int64_t targetId, const MixedProxyInfo& candidate)
  {
    // A collision must never borrow from itself:
    // (StrictlyUpperSameIndexPolicy should make this impossible, but we check nonetheless)
    if (targetId == candidate.sourceCollisionId)
      LOG(fatal) << "EventMixing: collision " << targetId << " offered itself as a mixing source.";
    const int32_t nSeen = ++candidateCount[targetSlot];
    std::uniform_int_distribution<int> pick(1, nSeen);
    if (pick(rng) == 1)
      lut[targetSlot] = candidate;
  }

  /// \brief How many different target collisions each source collision ended up supplying, from a finished LUT.
  /// \note  Keyed by source global index: sources are sparse, so a map is the right shape here even though the LUT is optimized as a vector.
  static std::unordered_map<int64_t, int> tallySourceUsage(std::vector<MixedProxyInfo> const& lut)
  {
    std::unordered_map<int64_t, int> usage;
    for (auto const& entry : lut) {
      if (entry.sourceCollisionId >= 0)
        usage[entry.sourceCollisionId]++;
    }
    return usage;
  }

  // Defining filters for events:
  Filter zvtxFilter = (nabs(o2::aod::lambdajetpol::zvtx) < maxZVtxPosition);

  // Preslices for correct collisions association:
  // (tested custom grouping and performs worse here)
  Preslice<aod::RingJets> perColJets = o2::aod::lambdajetpol::ringCollisionId;
  Preslice<aod::RingLaV0s> perColV0s = o2::aod::lambdajetpol::ringCollisionId;
  Preslice<aod::RingLeadPs> perColLeadPs = o2::aod::lambdajetpol::ringCollisionId;
  // // For doMixedEventProxies:
  // SliceCache mixCache;
  /// \brief Main analysis loop: for each collision, rebuilds the leading jet/particle proxies (with optional fakePolSwitches distortions), then loops over V0s computing the ring observable and polarization-vector profiles for every enabled kinematic-cut family.
  void processPolarizationData(soa::Filtered<o2::aod::RingCollisions> const& collisions, o2::aod::RingJets const& jets, o2::aod::RingLaV0s const& v0s,
                               o2::aod::RingLeadPs const& leadPs)
  {
    // Caches for borrowed physical jets (previous jet event mixing, or full window event mixing):
    ProxyCacheSlots prevJetCache{};    // Carried between collisions. Zero-initialized, so bools start as false
    ProxyCacheSlots mixedProxyCache{}; // Refilled per collision from the mixing LUTs
    // As forcePreviousJet and doMixedEventProxies are mutually exclusive we can bind them to a single variable:
    ProxyCacheSlots& proxyCache = fakePolSwitches.doMixedEventProxies ? mixedProxyCache : prevJetCache;
    // Neither should not be used along nProxyResamples > 1, as it does not apply to that case.
    // Ring definition for the whole run: full ring (default) or its longitudinal projection R_z = P_z n_z:
    const bool ringZMode = useRingZ;

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
    std::vector<JetProxyCache> jetProxyByCollision(collisionIndexExtent);
    std::vector<LeadPCache> leadPByCollision(collisionIndexExtent);
    std::vector<MixingAxes> mixingAxesByCollision(collisionIndexExtent);
    // doMixedEventProxies caches: three independent mixings, one per proxy (a collision may have a valid LeadP, but no SubLeadJet).
    // Mixing is binned on (Zvtx, proxy pt, centrality), varying only eta/phi.
    std::vector<MixedProxyInfo> mixedLeadPByCollision(collisionIndexExtent);
    std::vector<MixedProxyInfo> mixedLeadJetByCollision(collisionIndexExtent);
    std::vector<MixedProxyInfo> mixedSubJetByCollision(collisionIndexExtent);
    // Candidate counters for the the mixing: main loop below reads them back for QA.
    std::vector<int32_t> leadPCandidateCount(collisionIndexExtent, 0);
    std::vector<int32_t> leadJetCandidateCount(collisionIndexExtent, 0);
    std::vector<int32_t> subJetCandidateCount(collisionIndexExtent, 0);

    // Leading/subleading jet per collision, resolved once here instead of inside the (possibly nProxyResamples times resampled) main loop.
    // Both the main loop and the leadJet/subJet mixing functions read from this, so RingJets is scanned exactly once
    for (auto const& collision : collisions) {
      const auto collId = collision.globalIndex();
      const size_t slot = slotOf(collId);
      JetProxyCache cache;

      // Cached here so the pooling below never needs a collision iterator:
      mixingAxesByCollision[slot] = {collision.zvtx(), getCentrality(collision)};

      // std::optional avoids undefined behaviour from a default-constructed iterator:
      std::optional<o2::aod::RingJets::iterator> leadingJet;
      std::optional<o2::aod::RingJets::iterator> subleadingJet;
      for (auto const& jet : jets.sliceBy(perColJets, collId)) {
        const auto jetpt = jet.jetPt();
        if (jetpt > cache.leadingJetPt) {
          // Current leading becomes subleading:
          cache.subleadingJetPt = cache.leadingJetPt;
          subleadingJet = leadingJet; // may still be std::nullopt on first pass -- that is fine!
          // Now update the leading jet:
          cache.leadingJetPt = jetpt;
          leadingJet = jet;
        } else if (jetpt > cache.subleadingJetPt) { // Update subleading only:
          cache.subleadingJetPt = jetpt;
          subleadingJet = jet;
        }
      }
      // Finer control on jet momentum, further than TableProducer pre-selection:
      cache.hasValidLeadingJet = cache.leadingJetPt > minLeadJetPt;
      cache.hasValidSubJet = cache.subleadingJetPt > minSubLeadJetPt;
      if (cache.hasValidLeadingJet) {
        cache.leadingJetEta = leadingJet->jetEta();
        cache.leadingJetPhi = leadingJet->jetPhi();
      }
      if (cache.hasValidSubJet) {
        cache.subleadingJetEta = subleadingJet->jetEta();
        cache.subleadingJetPhi = subleadingJet->jetPhi();
      }
      jetProxyByCollision[slot] = cache;

      // The leading particle is resolved here too, so the pooling reads a vector instead of slicing RingLeadPs.
      // A collision with no leadP simply keeps isValid = false, and its mixing axes are stored regardless
      for (auto const& lp : leadPs.sliceBy(perColLeadPs, collId)) {
        leadPByCollision[slot] = {lp.leadParticlePt() > minLeadParticlePt, lp.leadParticlePt(), lp.leadParticleEta(), lp.leadParticlePhi()};
        break; // Table has at most one LeadP per collision, but we break nonetheless
      }
    }

    // A small guard:
    const bool doMixingQA = fakePolSwitches.doMixedEventProxies && qaSwitches.doEventMixingQA;

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
    // (This is performed out of the resampling loop, so nProxyResamples will not resample event mixing candidates)
    if (fakePolSwitches.doMixedEventProxies) {
      const auto tLutStart = std::chrono::steady_clock::now();

      // All three proxies bin on the same (Zvtx, proxy pT, centrality) axes, so a single policy object serves them, even though proxy pT varies:
      // We implement using methods straight from BinningPolicy.h, such as getBin() and getAllBinsCount()
      BinningPolicyBase<3> mixBinning{{axisConfigurations.axisPVz, axisConfigurations.axisJetPt, axisConfigurations.axisCentrality}, true}; // Ignore overflows (pT guards overflow to -999.f by construction)

      // Counting what actually enters the mixing (not what enters the dataframe), before any pooling starts:
      int64_t nValidLeadP = 0, nValidLeadJet = 0, nValidSubJet = 0;
      for (size_t slot = 0; slot < collisionIndexExtent; ++slot) {
        nValidLeadP += leadPByCollision[slot].isValid ? 1 : 0;
        nValidLeadJet += jetProxyByCollision[slot].hasValidLeadingJet ? 1 : 0;
        nValidSubJet += jetProxyByCollision[slot].hasValidSubJet ? 1 : 0;
      }
      LOG(info) << "LUT input: " << collisions.size() << " collisions, extent " << collisionIndexExtent << ", validLeadP " << nValidLeadP << ", validLeadJet " << nValidLeadJet << ", validSubJet " << nValidSubJet;

      /// \brief Aggregates every collision with a usable proxy by its mixing bin.
      ///        proxyPtOf returns the proxy pT for a slot, or a negative sentinel when that collision has no proxy this analysis would accept.
      /// \note  Collisions are visited in ascending index order, so each pool/window already comes out sorted by collision index.
      auto poolByBin = [&](auto const& proxyPtOf) {
        std::vector<std::vector<int32_t>> pools(mixBinning.getAllBinsCount()); // Method from BinningPolicy.h
        for (size_t slot = 0; slot < collisionIndexExtent; ++slot) {
          const float proxyPt = proxyPtOf(slot);
          if (proxyPt < 0.f)
            continue;
          auto const& axes = mixingAxesByCollision[slot];
          const int bin = mixBinning.getBin(std::make_tuple(axes.zvtx, proxyPt, axes.centrality));
          if (bin >= 0)
            pools[bin].push_back(static_cast<int32_t>(slot));
        }
        return pools;
      };

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
      const auto leadPPools = poolByBin([&](size_t slot) { return leadPByCollision[slot].isValid ? leadPByCollision[slot].pt : -999.f; });
      if (doMixingQA) {
        windowNeighboursOf(leadPPools, [&](size_t nNeighbours) { histos.fill(HIST("EventMixingQA/hMixedEventLeadPWindowNeighbours"), nNeighbours); });
      }
      const int64_t nPairsLeadP = pairWithinWindows(leadPPools, [&](size_t slot1, int64_t id1, size_t slot2, int64_t id2) {
        auto const& lp1 = leadPByCollision[slot1];
        auto const& lp2 = leadPByCollision[slot2];
        auto const& ax1 = mixingAxesByCollision[slot1];
        auto const& ax2 = mixingAxesByCollision[slot2];
        // Both sides of the pair feed each other's reservoir, properly using what is already in memory. This also lets a collision borrow from
        // earlier collisions as well, eliminating the forward-only bias of the sliding window in the event mixing procedure.
        reservoirInsert(leadPCandidateCount, mixedLeadPByCollision, slot1, id1, {lp2.pt, lp2.eta, lp2.phi, ax2.zvtx, ax2.centrality, id2});
        reservoirInsert(leadPCandidateCount, mixedLeadPByCollision, slot2, id2, {lp1.pt, lp1.eta, lp1.phi, ax1.zvtx, ax1.centrality, id1});
        if (doMixingQA)
          histos.fill(HIST("EventMixingQA/IndexQA/hMixedEventLeadPDeltaIndexEligible"), id2 - id1);
      });
      const auto tLeadPDone = std::chrono::steady_clock::now();
      LOG(info) << "  leadP loop: " << std::chrono::duration<double>(tLeadPDone - tLutStart).count() << " s, " << nPairsLeadP << " pairs";

      // Leading jet:
      const auto leadJetPools = poolByBin([&](size_t slot) { return jetProxyByCollision[slot].hasValidLeadingJet ? jetProxyByCollision[slot].leadingJetPt : -999.f; });
      if (doMixingQA) {
        windowNeighboursOf(leadJetPools, [&](size_t nNeighbours) { histos.fill(HIST("EventMixingQA/hMixedEventLeadJetWindowNeighbours"), nNeighbours); });
      }
      const int64_t nPairsLeadJet = pairWithinWindows(leadJetPools, [&](size_t slot1, int64_t id1, size_t slot2, int64_t id2) {
        auto const& lj1 = jetProxyByCollision[slot1];
        auto const& lj2 = jetProxyByCollision[slot2];
        auto const& ax1 = mixingAxesByCollision[slot1];
        auto const& ax2 = mixingAxesByCollision[slot2];
        reservoirInsert(leadJetCandidateCount, mixedLeadJetByCollision, slot1, id1, {lj2.leadingJetPt, lj2.leadingJetEta, lj2.leadingJetPhi, ax2.zvtx, ax2.centrality, id2});
        reservoirInsert(leadJetCandidateCount, mixedLeadJetByCollision, slot2, id2, {lj1.leadingJetPt, lj1.leadingJetEta, lj1.leadingJetPhi, ax1.zvtx, ax1.centrality, id1});
        if (doMixingQA)
          histos.fill(HIST("EventMixingQA/IndexQA/hMixedEventLeadJetDeltaIndexEligible"), id2 - id1);
      });
      const auto tLeadJetDone = std::chrono::steady_clock::now();
      LOG(info) << "  leadJet loop: " << std::chrono::duration<double>(tLeadJetDone - tLeadPDone).count() << " s, " << nPairsLeadJet << " pairs";

      // Subleading jet:
      const auto subJetPools = poolByBin([&](size_t slot) { return jetProxyByCollision[slot].hasValidSubJet ? jetProxyByCollision[slot].subleadingJetPt : -999.f; });
      if (doMixingQA) {
        windowNeighboursOf(subJetPools, [&](size_t nNeighbours) { histos.fill(HIST("EventMixingQA/hMixedEventSubJetWindowNeighbours"), nNeighbours); });
      }
      const int64_t nPairsSubJet = pairWithinWindows(subJetPools, [&](size_t slot1, int64_t id1, size_t slot2, int64_t id2) {
        auto const& sj1 = jetProxyByCollision[slot1];
        auto const& sj2 = jetProxyByCollision[slot2];
        auto const& ax1 = mixingAxesByCollision[slot1];
        auto const& ax2 = mixingAxesByCollision[slot2];
        reservoirInsert(subJetCandidateCount, mixedSubJetByCollision, slot1, id1, {sj2.subleadingJetPt, sj2.subleadingJetEta, sj2.subleadingJetPhi, ax2.zvtx, ax2.centrality, id2});
        reservoirInsert(subJetCandidateCount, mixedSubJetByCollision, slot2, id2, {sj1.subleadingJetPt, sj1.subleadingJetEta, sj1.subleadingJetPhi, ax1.zvtx, ax1.centrality, id1});
        if (doMixingQA)
          histos.fill(HIST("EventMixingQA/IndexQA/hMixedEventSubJetDeltaIndexEligible"), id2 - id1);
      });
      const auto tSubJetDone = std::chrono::steady_clock::now();
      LOG(info) << "  subJet loop: " << std::chrono::duration<double>(tSubJetDone - tLeadJetDone).count() << " s, " << nPairsSubJet << " pairs";


      // Source-usage QA:
      if (doMixingQA) {
        for (size_t slot = 0; slot < mixedLeadPByCollision.size(); ++slot) {
          auto const& entry = mixedLeadPByCollision[slot];
          if (entry.sourceCollisionId < 0) continue;
          histos.fill(HIST("EventMixingQA/IndexQA/hMixedEventLeadPDeltaIndexSelected"), (static_cast<int64_t>(slot) + collisionIndexBase) - entry.sourceCollisionId);
        }
        for (size_t slot = 0; slot < mixedLeadJetByCollision.size(); ++slot) {
          auto const& entry = mixedLeadJetByCollision[slot];
          if (entry.sourceCollisionId < 0) continue;
          histos.fill(HIST("EventMixingQA/IndexQA/hMixedEventLeadJetDeltaIndexSelected"), (static_cast<int64_t>(slot) + collisionIndexBase) - entry.sourceCollisionId);
        }
        for (size_t slot = 0; slot < mixedSubJetByCollision.size(); ++slot) {
          auto const& entry = mixedSubJetByCollision[slot];
          if (entry.sourceCollisionId < 0) continue;
          histos.fill(HIST("EventMixingQA/IndexQA/hMixedEventSubJetDeltaIndexSelected"), (static_cast<int64_t>(slot) + collisionIndexBase) - entry.sourceCollisionId);
        }

        auto leadPUsage = tallySourceUsage(mixedLeadPByCollision);
        for (auto const& kv : leadPUsage) histos.fill(HIST("EventMixingQA/SourceUsage/hMixedEventLeadPSourceUsageCount"), kv.second);
        for (auto const& entry : mixedLeadPByCollision) {
          if (entry.sourceCollisionId < 0) continue;
          auto usageIt = leadPUsage.find(entry.sourceCollisionId);
          if (usageIt == leadPUsage.end()) continue;
          histos.fill(HIST("EventMixingQA/SourceUsage/pMixedEventLeadPSourceUsageVsEta"), entry.eta, usageIt->second);
          histos.fill(HIST("EventMixingQA/SourceUsage/pMixedEventLeadPSourceUsageVsPhi"), entry.phi, usageIt->second);
          leadPUsage.erase(usageIt); 
        }

        auto leadJetUsage = tallySourceUsage(mixedLeadJetByCollision);
        for (auto const& kv : leadJetUsage) histos.fill(HIST("EventMixingQA/SourceUsage/hMixedEventLeadJetSourceUsageCount"), kv.second);
        for (auto const& entry : mixedLeadJetByCollision) {
          if (entry.sourceCollisionId < 0) continue;
          auto usageIt = leadJetUsage.find(entry.sourceCollisionId);
          if (usageIt == leadJetUsage.end()) continue;
          histos.fill(HIST("EventMixingQA/SourceUsage/pMixedEventLeadJetSourceUsageVsEta"), entry.eta, usageIt->second);
          histos.fill(HIST("EventMixingQA/SourceUsage/pMixedEventLeadJetSourceUsageVsPhi"), entry.phi, usageIt->second);
          leadJetUsage.erase(usageIt);
        }

        auto subJetUsage = tallySourceUsage(mixedSubJetByCollision);
        for (auto const& kv : subJetUsage) histos.fill(HIST("EventMixingQA/SourceUsage/hMixedEventSubJetSourceUsageCount"), kv.second);
        for (auto const& entry : mixedSubJetByCollision) {
          if (entry.sourceCollisionId < 0) continue;
          auto usageIt = subJetUsage.find(entry.sourceCollisionId);
          if (usageIt == subJetUsage.end()) continue;
          histos.fill(HIST("EventMixingQA/SourceUsage/pMixedEventSubJetSourceUsageVsEta"), entry.eta, usageIt->second);
          histos.fill(HIST("EventMixingQA/SourceUsage/pMixedEventSubJetSourceUsageVsPhi"), entry.phi, usageIt->second);
          subJetUsage.erase(usageIt);
        }
      }
      LOG(info) << "LUT build: " << std::chrono::duration<double>(std::chrono::steady_clock::now() - tLutStart).count() << " s for " << collisions.size() << " collisions";
    }

    for (int idxResampling = 0; idxResampling < fakePolSwitches.nProxyResamples; idxResampling++) { // resampling loop for forceRandJet and forceDatalikeJet
      for (auto const& collision : collisions) {
        if (fakePolSwitches.doMixedEventProxies) { // Making sure cache is properly reset for each new collision
          proxyCache = {};
        }
        const float collisionPVz = collision.zvtx();

        const auto collId = collision.globalIndex(); // The self-index accessor
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
        float leadPPt = -1.; // pT = -1 means "table entry not found for this collision".
        float leadPEta = 0.;
        float leadPPhi = 0.;
        float leadPPx = 0., leadPPy = 0., leadPPz = 0.;
        for (auto const& lp : leadPsInColl) {
          // Table should contain exactly one entry per collision, but we break immediately to be safe:
          leadPPt = lp.leadParticlePt();
          leadPEta = lp.leadParticleEta();
          leadPPhi = lp.leadParticlePhi();
          // Using dynamic columns to make code cleaner:
          leadPPx = lp.leadParticlePx();
          leadPPy = lp.leadParticlePy();
          leadPPz = lp.leadParticlePz();
        }
        // // Discard events with no leading particle (FastJet didn't even run in these cases!):
        // if (leadPPt < 0.)
        //   continue;

        // Apply minimum pT selection for the leading particle (not necessarily the same as in derived data builder. Can be a stricter cut!):
        bool hasValidLeadingP = leadPPt > minLeadParticlePt;

        // Snapshot this collision's own leading particle before any distortion can overwrite it,
        // so forcePreviousJet can hand it to the next collision:
        const bool beforeDistHasValidLeadP = hasValidLeadingP;
        const float beforeDistLeadPPt = leadPPt;
        const float beforeDistLeadPEta = leadPEta;
        const float beforeDistLeadPPhi = leadPPhi;

        // Build leading particle unit vector, outside the V0 loop for performance.
        XYZVector leadPUnitVec(1., 0., 0.); // dummy (overwritten below when hasValidLeadingP)
        float leadPZ = 0.;                  // \hat t_z of the leading-particle proxy (see axisLeadPZ)
        if (hasValidLeadingP) {
          leadPUnitVec = XYZVector(leadPPx, leadPPy, leadPPz).Unit();
          leadPZ = leadPUnitVec.Z();

          // doMixedEventProxies: get this collision's mixing partner (if any) from the LUT built above:
          // (this check is performed only if hasValidLeadingP is true for performance, as the LeadPPt matching for the event mixing would also demand a minimum jet pT)
          // (it is also a physics selection: we want to mix the correlations in events that could actually have a ring formed)
          if (fakePolSwitches.doMixedEventProxies) {
            auto const& mixLeadP = mixedLeadPByCollision[slotOf(collId)];
            proxyCache.hadLeadP = (mixLeadP.sourceCollisionId >= 0); // Empty slot means this event had no valid mixing target

            // Declaring bools before the hadLeadP checks as they will be used inside the doMixingQA block too:
            bool samePtProxyLeadP = false;
            bool sameEtaProxyLeadP = false;
            bool samePhiProxyLeadP = false;
            if (proxyCache.hadLeadP) {
              proxyCache.leadPPt = mixLeadP.pt;
              proxyCache.leadPEta = mixLeadP.eta;
              proxyCache.leadPPhi = mixLeadP.phi;

              // bit-by-bit checks of correspondence (asked for no-lint on these "paranoid" checks):
              // (inexpensive enough that we can check this at production time)
              samePtProxyLeadP = (mixLeadP.pt == leadPPt);    // NOLINT(clang-diagnostic-float-equal)
              sameEtaProxyLeadP = (mixLeadP.eta == leadPEta); // NOLINT(clang-diagnostic-float-equal)
              samePhiProxyLeadP = (mixLeadP.phi == leadPPhi); // NOLINT(clang-diagnostic-float-equal)
              if (samePtProxyLeadP && sameEtaProxyLeadP && samePhiProxyLeadP) { // Astronomically small chance, even with the birthday paradox
                LOG(fatal) << "EventMixing: borrowed leadP is bit-identical to the target's own in pt, eta and phi. Target collision "
                           << collId << ", source " << mixLeadP.sourceCollisionId << ", pt " << leadPPt;
              }
            }

            if (doMixingQA) {
              // Filled hit or miss, so the zero bin is the miss rate seen from the pool's side:
              const int nCandidates = leadPCandidateCount[slotOf(collId)]; // Zero-initialised, so no existence check needed
              histos.fill(HIST("EventMixingQA/CollLoopOutcome/hMixedEventLeadPCandidates"), nCandidates);
              histos.fill(HIST("EventMixingQA/CollLoopOutcome/h2dMixedLeadPCandidatesVsPt"), nCandidates, leadPPt);

              if (proxyCache.hadLeadP) {
                // Only fill the comparison histograms on an actual hit (source vs. this collision's own):
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/hMixedEventLeadPOutcome"), 1);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/h2dMixedEventLeadPOutcomeVsTargetZVtx"), 1, collisionPVz);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/h2dMixedEventLeadPOutcomeVsTargetCentrality"), 1, centrality);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/h2dMixedEventLeadPOutcomeVsTargetProxyPt"), 1, leadPPt);

                histos.fill(HIST("EventMixingQA/AnglrCorrltns/h2dMixedLeadPEtaVsLeadPEta"), proxyCache.leadPEta, leadPEta);
                histos.fill(HIST("EventMixingQA/AnglrCorrltns/h2dMixedLeadPPhiVsLeadPPhi"), proxyCache.leadPPhi, leadPPhi);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/hMixedEventLeadPDeltaPt"), proxyCache.leadPPt - leadPPt);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/hMixedEventLeadPDeltaZvtx"), mixLeadP.zvtx - collisionPVz);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/hMixedEventLeadPDeltaCent"), mixLeadP.centrality - centrality);

                // QAing the binning policy itself to verify if any wrong proxies are being borrowed:
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/h2dMixedEventLeadPDeltaPtVsTargetPt"), proxyCache.leadPPt - leadPPt, leadPPt);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/h2dMixedEventLeadPDeltaZvtxVsTargetZvtx"), mixLeadP.zvtx - collisionPVz, collisionPVz);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/h2dMixedEventLeadPDeltaCentVsTargetCent"), mixLeadP.centrality - centrality, centrality);

                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/hMixedEventLeadPDeltaEta"), proxyCache.leadPEta - leadPEta);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/hMixedEventLeadPDeltaPhi"), wrapToPiFast(proxyCache.leadPPhi - leadPPhi));

                // 2D Index Vs Delta Mixed-Variable QAs:
                const float deltaIndexLeadP = static_cast<float>(collId - mixLeadP.sourceCollisionId);
                histos.fill(HIST("EventMixingQA/IndexQA/h2dMixedEventLeadPDeltaPtVsDeltaIndex"), deltaIndexLeadP, mixLeadP.pt - leadPPt);
                histos.fill(HIST("EventMixingQA/IndexQA/h2dMixedEventLeadPDeltaZvtxVsDeltaIndex"), deltaIndexLeadP, mixLeadP.zvtx - collisionPVz);
                histos.fill(HIST("EventMixingQA/IndexQA/h2dMixedEventLeadPDeltaCentVsDeltaIndex"), deltaIndexLeadP, mixLeadP.centrality - centrality);

                // Computing final needed bool, which can be declared as a const:
                const bool sameZvtxLeadP = (mixLeadP.zvtx == collisionPVz); // NOLINT(clang-diagnostic-float-equal)

                if (sameZvtxLeadP && samePtProxyLeadP) // Flagged as "alarm" level
                  LOG(alarm) << "EventMixing: source and target share Zvtx and leadP pt bit-for-bit -- duplicated collision row? Target "
                             << collId << ", source " << mixLeadP.sourceCollisionId << ", Zvtx " << collisionPVz;

                // Single-variable coincidences are possible by chance, so they are counted instead:
                if (samePtProxyLeadP)
                  histos.fill(HIST("EventMixingQA/IdentityChecks/hMixedEventLeadPIdentityFlags"), 0);
                if (sameEtaProxyLeadP)
                  histos.fill(HIST("EventMixingQA/IdentityChecks/hMixedEventLeadPIdentityFlags"), 1);
                if (samePhiProxyLeadP)
                  histos.fill(HIST("EventMixingQA/IdentityChecks/hMixedEventLeadPIdentityFlags"), 2);
                if (sameZvtxLeadP)
                  histos.fill(HIST("EventMixingQA/IdentityChecks/hMixedEventLeadPIdentityFlags"), 3);

                // Possible shared-track case in event splitting, done in a log scale:
                const float angularSepLeadP = std::hypot(mixLeadP.eta - leadPEta, wrapToPiFast(mixLeadP.phi - leadPPhi));
                histos.fill(HIST("EventMixingQA/IdentityChecks/hMixedEventLeadPLogAngularSep"), angularSepLeadP > 0.f ? std::log10(angularSepLeadP) : -7.f);
              } else {
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/hMixedEventLeadPOutcome"), 0);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/h2dMixedEventLeadPOutcomeVsTargetZVtx"), 0, collisionPVz);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/h2dMixedEventLeadPOutcomeVsTargetCentrality"), 0, centrality);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/h2dMixedEventLeadPOutcomeVsTargetProxyPt"), 0, leadPPt);
              }
            }
          }

          // Apply distortion logic:
          // (modifies (leadPPt, leadPEta, leadPPhi, leadPUnitVec) as if the modified proxy was the actual proxy of this event)
          applyProxyDistortion({hasValidLeadingP, leadPPt, leadPEta, leadPPhi, leadPUnitVec},
                               minLeadParticlePt, fakePolSwitches.maxLeadPProxyEta, {proxyCache.hadLeadP, proxyCache.leadPPt, proxyCache.leadPEta, proxyCache.leadPPhi},
                               etaLeadPDist, phiLeadPDist);

          // Fill distorted-proxy QA histograms (i.e., the actually used proxy).
          // A borrowed-proxy miss and a failed pT gate will skip the fill.
          if (hasValidLeadingP) {
            histos.fill(HIST("KinematicsQA/Jet/hLeadPEta"), leadPEta);
            histos.fill(HIST("KinematicsQA/Jet/hLeadPPhi"), leadPPhi);
            histos.fill(HIST("KinematicsQA/Jet/hJetCounterPtLeadP"), leadPPt);

            histos.fill(HIST("KinematicsQA/Jet/h2dLeadPEtaVsPhi"), leadPEta, leadPPhi);
            histos.fill(HIST("KinematicsQA/Jet/h2dLeadPEtaVsPVz"), leadPEta, collisionPVz);
            // if (doJetProxy5dQA)
            //   histos.fill(HIST("KinematicsQA/Jet/h5dLeadPEtaPhiPtPVzCent"), leadPEta, leadPPhi, leadPPt, collisionPVz, centrality);
          }
        }

        // 3) Fetching leading jet and subleading jet -- Resolved once per collision in the jetProxyByCollision pre-pass:
        const JetProxyCache& jetProxies = jetProxyByCollision[slotOf(collId)];
        float leadingJetPt = jetProxies.leadingJetPt;
        float subleadingJetPt = jetProxies.subleadingJetPt;

        // Defining local bools that may be changed by applyProxyDistortion:
        bool hasValidLeadingJet = jetProxies.hasValidLeadingJet;
        bool hasValidSubJet = jetProxies.hasValidSubJet;

        // Build jet vectors (only when the corresponding jet exists):
        // Dummy initialisations are safe: all jet-dependent fills are gated on hasValidLeadingJet / hasValidSubJet.
        float leadingJetEta = 0.;
        float leadingJetPhi = 0.;
        // Useful scalars for PrimeJet coordinates:
        float jetZ = 0.;
        float inverseJetTransverse = 0.;
        XYZVector leadingJetUnitVec(1., 0., 0.); // dummy (overwritten below)
        if (hasValidLeadingJet) {
          leadingJetEta = jetProxies.leadingJetEta;
          leadingJetPhi = jetProxies.leadingJetPhi;
          // Rebuild the direction from the cached eta/phi (cheaper than calling jetPx() internal getters and then normalizing with .Unit()):
          const double inverseCoshEta = 1.0 / std::cosh(leadingJetEta);
          leadingJetUnitVec = XYZVector(std::cos(leadingJetPhi) * inverseCoshEta, std::sin(leadingJetPhi) * inverseCoshEta, std::tanh(leadingJetEta));

          // Apply distortion logic:
          if (fakePolSwitches.doMixedEventProxies) {
            // Get this collision's leading-jet mixing partner (if any) from the LUT built above:
            auto const& mixLeadJet = mixedLeadJetByCollision[slotOf(collId)];
            proxyCache.hadLeadJet = (mixLeadJet.sourceCollisionId >= 0);
            if (proxyCache.hadLeadJet) {
              proxyCache.leadJetPt = mixLeadJet.pt;
              proxyCache.leadJetEta = mixLeadJet.eta;
              proxyCache.leadJetPhi = mixLeadJet.phi;

              // bit-by-bit checks of correspondence:
              const bool samePtProxyLeadJet = (mixLeadJet.pt == leadingJetPt);      // NOLINT(clang-diagnostic-float-equal)
              const bool sameEtaProxyLeadJet = (mixLeadJet.eta == leadingJetEta);   // NOLINT(clang-diagnostic-float-equal)
              const bool samePhiProxyLeadJet = (mixLeadJet.phi == leadingJetPhi);   // NOLINT(clang-diagnostic-float-equal)
              if (samePtProxyLeadJet && sameEtaProxyLeadJet && samePhiProxyLeadJet) { // Astronomically small chance, even with the birthday paradox
                LOG(fatal) << "EventMixing: borrowed LeadJet is bit-identical to the target's own in pt, eta and phi. Target collision "
                           << collId << ", source " << mixLeadJet.sourceCollisionId << ", pt " << leadingJetPt;
              }
            }

            if (doMixingQA) {
              const int nCandidates = leadJetCandidateCount[slotOf(collId)];
              histos.fill(HIST("EventMixingQA/CollLoopOutcome/hMixedEventLeadJetCandidates"), nCandidates);
              histos.fill(HIST("EventMixingQA/CollLoopOutcome/h2dMixedLeadJetCandidatesVsPt"), nCandidates, leadingJetPt);

              if (proxyCache.hadLeadJet) {
                // Only fill the comparison histograms on an actual hit (source vs. this collision's own):
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/hMixedEventLeadJetOutcome"), 1);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/h2dMixedEventLeadJetOutcomeVsTargetZVtx"), 1, collisionPVz);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/h2dMixedEventLeadJetOutcomeVsTargetCentrality"), 1, centrality);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/h2dMixedEventLeadJetOutcomeVsTargetProxyPt"), 1, leadingJetPt);

                histos.fill(HIST("EventMixingQA/AnglrCorrltns/h2dMixedLeadJetEtaVsLeadJetEta"), proxyCache.leadJetEta, leadingJetEta);
                histos.fill(HIST("EventMixingQA/AnglrCorrltns/h2dMixedLeadJetPhiVsLeadJetPhi"), proxyCache.leadJetPhi, leadingJetPhi);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/hMixedEventLeadJetDeltaPt"), proxyCache.leadJetPt - leadingJetPt);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/hMixedEventLeadJetDeltaZvtx"), mixLeadJet.zvtx - collisionPVz);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/hMixedEventLeadJetDeltaCent"), mixLeadJet.centrality - centrality);

                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/hMixedEventLeadJetDeltaEta"), proxyCache.leadJetEta - leadingJetEta);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/hMixedEventLeadJetDeltaPhi"), wrapToPiFast(proxyCache.leadJetPhi - leadingJetPhi));
              } else {
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/hMixedEventLeadJetOutcome"), 0);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/h2dMixedEventLeadJetOutcomeVsTargetZVtx"), 0, collisionPVz);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/h2dMixedEventLeadJetOutcomeVsTargetCentrality"), 0, centrality);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/h2dMixedEventLeadJetOutcomeVsTargetProxyPt"), 0, leadingJetPt);
              }
            }
          }
          applyProxyDistortion({hasValidLeadingJet, leadingJetPt, leadingJetEta, leadingJetPhi, leadingJetUnitVec},
                               minLeadJetPt, fakePolSwitches.maxJetProxyEta, {proxyCache.hadLeadJet, proxyCache.leadJetPt, proxyCache.leadJetEta, proxyCache.leadJetPhi},
                               etaLeadPDist, phiLeadPDist);

          // Fill distorted-proxy QA histograms:
          // Do not gate on the post-distortion hasValidLeadingJet (a pT cut) value here!
          if (hasValidLeadingJet) {
            histos.fill(HIST("KinematicsQA/Jet/hLeadJetEta"), leadingJetEta);
            histos.fill(HIST("KinematicsQA/Jet/hLeadJetPhi"), leadingJetPhi);
            histos.fill(HIST("KinematicsQA/Jet/hJetCounterPtJet"), leadingJetPt);

            histos.fill(HIST("KinematicsQA/Jet/h2dLeadJetEtaVsPhi"), leadingJetEta, leadingJetPhi);
            histos.fill(HIST("KinematicsQA/Jet/h2dLeadJetEtaVsPVz"), leadingJetEta, collisionPVz);
            // if (doJetProxy5dQA)
            //   histos.fill(HIST("KinematicsQA/Jet/h5dLeadJetEtaPhiPtPVzCent"), leadingJetEta, leadingJetPhi, leadingJetPt, collisionPVz, centrality);

            // PrimeJet coordinate system variables:
            // (Calculated once per collision, after applyProxyDistortion, using a Gram-Schmidt-like orthonormalisation)
            // \hat x_Jet = (\hat z - t_z \hat t)/t_T, \hat y_Jet = (\hat t \times \hat z)/t_T
            jetZ = leadingJetUnitVec.Z(); // \hat z_Jet = \hat t
            const float jetFrameTt = std::sqrt(1.0f - jetZ * jetZ); // t_T = sqrt(1 - t_z^2) = 1/cosh(eta_jet)
            inverseJetTransverse = (jetFrameTt > 1e-4f) ? (1.0f / jetFrameTt) : 0.0f; // Paranoid guard with the pseudorapidity cuts, but safer nonetheless
          }
        }

        float subleadingJetEta = 0.;
        float subleadingJetPhi = 0.;
        XYZVector subJetUnitVec(1., 0., 0.);
        if (hasValidSubJet) {
          subleadingJetEta = jetProxies.subleadingJetEta;
          subleadingJetPhi = jetProxies.subleadingJetPhi;
          const double inverseCoshEtaSub = 1.0 / std::cosh(subleadingJetEta);
          subJetUnitVec = XYZVector(std::cos(subleadingJetPhi) * inverseCoshEtaSub, std::sin(subleadingJetPhi) * inverseCoshEtaSub, std::tanh(subleadingJetEta));

          // Apply distortion logic:
          if (fakePolSwitches.doMixedEventProxies) {
            // Get this collision's subleading-jet mixing partner (if any) from the LUT built above:
            auto const& mixSubJet = mixedSubJetByCollision[slotOf(collId)];
            proxyCache.hadSubJet = (mixSubJet.sourceCollisionId >= 0);
            if (proxyCache.hadSubJet) {
              proxyCache.subJetPt = mixSubJet.pt;
              proxyCache.subJetEta = mixSubJet.eta;
              proxyCache.subJetPhi = mixSubJet.phi;

              // bit-by-bit checks of correspondence:
              const bool samePtProxySubJet = (mixSubJet.pt == subleadingJetPt);      // NOLINT(clang-diagnostic-float-equal)
              const bool sameEtaProxySubJet = (mixSubJet.eta == subleadingJetEta);   // NOLINT(clang-diagnostic-float-equal)
              const bool samePhiProxySubJet = (mixSubJet.phi == subleadingJetPhi);   // NOLINT(clang-diagnostic-float-equal)
              if (samePtProxySubJet && sameEtaProxySubJet && samePhiProxySubJet) { // Astronomically small chance, even with the birthday paradox
                LOG(fatal) << "EventMixing: borrowed SubJet is bit-identical to the target's own in pt, eta and phi. Target collision "
                           << collId << ", source " << mixSubJet.sourceCollisionId << ", pt " << subleadingJetPt;
              }
            }

            if (doMixingQA) {
              const int nCandidates = subJetCandidateCount[slotOf(collId)];
              histos.fill(HIST("EventMixingQA/CollLoopOutcome/hMixedEventSubJetCandidates"), nCandidates);
              histos.fill(HIST("EventMixingQA/CollLoopOutcome/h2dMixedSubJetCandidatesVsPt"), nCandidates, subleadingJetPt);

              if (proxyCache.hadSubJet) {
                // Only fill the comparison histograms on an actual hit (source vs. this collision's own):
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/hMixedEventSubJetOutcome"), 1);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/h2dMixedEventSubJetOutcomeVsTargetZVtx"), 1, collisionPVz);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/h2dMixedEventSubJetOutcomeVsTargetCentrality"), 1, centrality);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/h2dMixedEventSubJetOutcomeVsTargetProxyPt"), 1, subleadingJetPt);

                histos.fill(HIST("EventMixingQA/AnglrCorrltns/h2dMixedSubJetEtaVsSubJetEta"), proxyCache.subJetEta, subleadingJetEta);
                histos.fill(HIST("EventMixingQA/AnglrCorrltns/h2dMixedSubJetPhiVsSubJetPhi"), proxyCache.subJetPhi, subleadingJetPhi);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/hMixedEventSubJetDeltaPt"), proxyCache.subJetPt - subleadingJetPt);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/hMixedEventSubJetDeltaZvtx"), mixSubJet.zvtx - collisionPVz);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/hMixedEventSubJetDeltaCent"), mixSubJet.centrality - centrality);

                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/hMixedEventSubJetDeltaEta"), proxyCache.subJetEta - subleadingJetEta);
                histos.fill(HIST("EventMixingQA/SourceTargetDeltas/hMixedEventSubJetDeltaPhi"), wrapToPiFast(proxyCache.subJetPhi - subleadingJetPhi));
              } else {
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/hMixedEventSubJetOutcome"), 0);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/h2dMixedEventSubJetOutcomeVsTargetZVtx"), 0, collisionPVz);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/h2dMixedEventSubJetOutcomeVsTargetCentrality"), 0, centrality);
                histos.fill(HIST("EventMixingQA/CollLoopOutcome/h2dMixedEventSubJetOutcomeVsTargetProxyPt"), 0, subleadingJetPt);
              }
            }
          }
          applyProxyDistortion({hasValidSubJet, subleadingJetPt, subleadingJetEta, subleadingJetPhi, subJetUnitVec},
                               minSubLeadJetPt, fakePolSwitches.maxJetProxyEta, {proxyCache.hadSubJet, proxyCache.subJetPt, proxyCache.subJetEta, proxyCache.subJetPhi},
                               etaLeadPDist, phiLeadPDist);

          // Fill distorted-proxy QA histograms:
          // Do not gate on the post-distortion hasValidSubJet (a pT cut) value here!
          if (hasValidSubJet) {
            histos.fill(HIST("KinematicsQA/Jet/hSubLeadJetEta"), subleadingJetEta);
            histos.fill(HIST("KinematicsQA/Jet/hSubLeadJetPhi"), subleadingJetPhi);
            histos.fill(HIST("KinematicsQA/Jet/hJetCounterPt2ndJet"), subleadingJetPt);

            histos.fill(HIST("KinematicsQA/Jet/h2dSubLeadJetEtaVsPhi"), subleadingJetEta, subleadingJetPhi);
            histos.fill(HIST("KinematicsQA/Jet/h2dSubLeadJetEtaVsPVz"), subleadingJetEta, collisionPVz);
            // if (doJetProxy5dQA)
            //   histos.fill(HIST("KinematicsQA/Jet/h5dSubLeadJetEtaPhiPtPVzCent"), subleadingJetEta, subleadingJetPhi, subleadingJetPt, collisionPVz, centrality);
          }
        }

        // forcePreviousJet: hand this collision's own proxies to the next collision.
        if (fakePolSwitches.forcePreviousJet) {
          prevJetCache = ProxyCacheSlots{jetProxies.hasValidLeadingJet, jetProxies.leadingJetPt, jetProxies.leadingJetEta, jetProxies.leadingJetPhi,
                                         jetProxies.hasValidSubJet, jetProxies.subleadingJetPt, jetProxies.subleadingJetEta, jetProxies.subleadingJetPhi,
                                         beforeDistHasValidLeadP, beforeDistLeadPPt, beforeDistLeadPEta, beforeDistLeadPPhi};
        }

        // (jet eta cuts only meaningful when the jet actually exists)
        const bool kinematicJetCheck = hasValidLeadingJet && (std::abs(leadingJetEta) < 0.5);
        const bool kinematic2ndJetCheck = hasValidSubJet && (std::abs(subleadingJetEta) < 0.5);
        const bool kinematicLeadPCheck = hasValidLeadingP && (std::abs(leadPEta) < 0.5);

        // Quick bools that are useful for detector asymmetry QA:
        const bool jetEtaPos = hasValidLeadingJet && (leadingJetEta >= 0.); // Only perform >= check if has validJet
        const bool subJetEtaPos = hasValidSubJet && (subleadingJetEta >= 0.);
        const bool leadPEtaPos = hasValidLeadingP && (leadPEta >= 0.);

        // Stricter QA version of the bools -- Jets have a radius that makes it possible eta_{jet} > 0, yet half its tracks are in eta < 0
        // (This does not apply to leading particles, obviously. They have no substructure in eta)
        const bool jetEtaStrict = hasValidLeadingJet && (std::abs(leadingJetEta) >= jetR);
        const bool subJetEtaStrict = hasValidSubJet && (std::abs(subleadingJetEta) >= jetR);
        // If one was to define bools for each side of the detector (not needed in the current if-else structure on TProfile fills)
        // const bool jetEtaStrictPos = jetEtaPos && jetEtaStrict;
        // const bool jetEtaStrictNeg = !jetEtaPos && jetEtaStrict;
        // const bool subJetEtaStrictPos = subJetEtaPos && subJetEtaStrict;
        // const bool subJetEtaStrictNeg = !subJetEtaPos && subJetEtaStrict;

        // Fetching number of Lambda-like V0s in collision (must be known before full loop, to fill "pRingVsNV0s"):
        // int nLambdaLikeV0s = 0;
        // for (auto const& v0 : v0sInColl) {
        //   if (v0.isLambda() ^ v0.isAntiLambda()){ // XOR (only the non-ambiguous candidates)
        //     nLambdaLikeV0s++;
        //   }
        // }
        // Code above was superseeded: new datamodel does not store ambiguous candidates!
        // The new getter comes at the very start of processPolarizationData() now.

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
          if (hasValidLeadingJet) { // Using post-distortion/mixing variables for the jets
            histos.fill(HIST("KinematicsQA/V0/h2dV0PhiVsJetPhi"), v0phi, leadingJetPhi);
            histos.fill(HIST("KinematicsQA/V0/h2dV0EtaVsJetEta"), v0eta, leadingJetEta);
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

          // Calculating the phi* angle:
          // e_z = p_Lambda_hat; // e_x = normalize(z_hat cross p_Lambda); // e_y = e_z cross e_x;
          // // phi_star = atan2(p_p_star dot e_y, p_p_star dot e_x);
          // XYZVector e_x(-lambdaLikeUnit3Vec.Y(), lambdaLikeUnit3Vec.X(), 0.); // Same as e_x = zHat.Cross(lambdaLikeUnit3Vec);
          // XYZVector e_y = lambdaLikeUnit3Vec.Cross(e_x); // e_y completes the right-handed coordinate system (e_z is lambdaLikeUnit3Vec)
          // float pX = protonLikeStarUnit3Vec.Dot(e_x);
          // float pY = protonLikeStarUnit3Vec.Dot(e_y);
          // float phiStar = std::atan2(pY, pX);
          // Faster implementation:
          // pX = p_y * L_x - p_x * L_y
          float pX = protonLikeStarUnit3Vec.Y() * lambdaLikeUnit3Vec.X() - protonLikeStarUnit3Vec.X() * lambdaLikeUnit3Vec.Y();
          // pY = p_z - L_z * (p_proton_star dot p_lambda_hat)
          float pY = protonLikeStarUnit3Vec.Z() - lambdaLikeUnit3Vec.Z() * cosFakePol; // (Reusing cosFakePol calculated earlier!)
          float phiStar = std::atan2(pY, pX);                                          // This will give an output from -PI to PI

          // Ring prefactor: same as polPrefactor, but forcePolSignQA inverts it for antiLambdas only
          const float polPrefactor = isLambda ? PolPrefactorLambda : PolPrefactorAntiLambda;
          const float ringPrefactor = (fakePolSwitches.forcePolSignQA && !isLambda) ? -polPrefactor : polPrefactor;
          const float v0p = ringZMode ? lambdaLike3Vec.R() : 0.f; // |p_Lambda|, only used for the R_z bearing chi

          // Calculating polarization observables (in the Lambda frame, because that is easier -- does not require boosts):
          // To be precise, not the polarization itself, but a part of the summand in P^*_Lambda = (3/\alpha_Lambda) * <p^*_{proton}>
          const float polStarX = polPrefactor * protonLikeStarUnit3Vec.X();
          const float polStarY = polPrefactor * protonLikeStarUnit3Vec.Y();
          const float polStarZ = polPrefactor * protonLikeStarUnit3Vec.Z();

          // Calculating rotated coordinate systems:
          // AEE-frame's relevant polarization:
          const float protonStarPt = protonLikeStarUnit3Vec.Rho(); // TODO: check if this makes sense. Should it be the un-normalized version instead?

          // Guarded inverse values for calculating rotations, similar to the jetZ guard below:
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
              histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/hLambdaMassVsPhiLambdaMinusPhiProtonStar"), v0LambdaLikeMass, deltaPhiLambdaProtonStar);
            else
              histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/hAntiLambdaMassVsPhiLambdaMinusPhiProtonStar"), v0LambdaLikeMass, deltaPhiLambdaProtonStar);
            // AEE and HEE correlation:
            histos.fill(HIST("HelicityEfficiencyQA/hFakePolCounts_CosThetaVsPhiStar"), cosFakePol, phiStar);
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

          ////////////////////////////////////////////
          // Ring observable: Leading particle proxy
          // Only computed when a valid leading particle exists (pT > minLeadParticlePt)
          ////////////////////////////////////////////
          float ringObservableLeadP = 0.;
          float deltaPhiLeadP = 0.;
          float deltaThetaLeadP = 0.;
          float cosDeltaThetaLeadP = 0.;
          // KappaEff moments (unitless ring squared and its projection weight) and the R_z-only diagnostics:
          float kappaNumLeadP = 0.;
          float kappaDenLeadP = 1.;
          float ringPerpLeadP = 0.;
          float chiLeadP = 0.;
          if (hasValidLeadingP) {
            XYZVector crossLeadP = leadPUnitVec.Cross(lambdaLike3Vec);
            const float invCrossNormLeadP = 1.f / crossLeadP.R(); // Caching the .R() result
            if (!ringZMode) {
              ringObservableLeadP = protonLikeStarUnit3Vec.Dot(crossLeadP) * invCrossNormLeadP;
            } else {
              const float nzLeadP = crossLeadP.Z() * invCrossNormLeadP;
              ringObservableLeadP = protonLikeStarUnit3Vec.Z() * nzLeadP;
              kappaDenLeadP = nzLeadP * nzLeadP;
              // Diagnostics only: the full ring (for R_perp) and the bearing chi:
              ringPerpLeadP = ringPrefactor * (protonLikeStarUnit3Vec.Dot(crossLeadP) * invCrossNormLeadP - ringObservableLeadP);
              chiLeadP = std::atan2(-crossLeadP.Z() * v0p, crossLeadP.Y() * v0px - crossLeadP.X() * v0py);
            }
            kappaNumLeadP = ringObservableLeadP * ringObservableLeadP;
            // Adding the prefactor related to the CP-violating decay (decay constants have different signs)
            ringObservableLeadP *= ringPrefactor;
            // Angular variables
            deltaPhiLeadP = wrapToPiFast(v0phi - leadPPhi); // Wrapped to [-PI, PI), for convenience

            cosDeltaThetaLeadP = leadPUnitVec.Dot(lambdaLikeUnit3Vec); // Uses the pre-calculated unit vectors to avoid recomputation
            deltaThetaLeadP = std::acos(cosDeltaThetaLeadP);           // 3D angular separation. Same as ROOT::Math::VectorUtil::Angle(leadPUnitVec, lambdaLike3Vec);
          }

          //////////////////////////////////////////
          // Ring observable: Leading jet proxy
          // Only computed when a leading jet exists in this collision.
          //////////////////////////////////////////
          float ringObservable = 0.;
          float deltaPhiJet = 0.;
          float deltaEtaJet = 0.;
          float deltaThetaJet = 0.;
          float cosDeltaThetaJet = 0.;
          float ringObservableOverJetZ = 0.;
          // KappaEff moments and the R_z-only diagnostics:
          float kappaNumJet = 0.;
          float kappaDenJet = 1.;
          float ringPerpJet = 0.;
          float chiJet = 0.;
          // PrimeJet-frame components:
          float polStarXPrimeJet = 0.;
          float polStarYPrimeJet = 0.;
          float polStarZPrimeJet = 0.;
          float v0pxPrimeJet = 0.;
          float v0pyPrimeJet = 0.;
          float v0pzPrimeJet = 0.;
          if (hasValidLeadingJet) {
            XYZVector cross = leadingJetUnitVec.Cross(lambdaLike3Vec);
            const float invCrossNorm = 1.f / cross.R();
            if (!ringZMode) {
              ringObservable = protonLikeStarUnit3Vec.Dot(cross) * invCrossNorm;
            } else {
              const float nzJet = cross.Z() * invCrossNorm;
              ringObservable = protonLikeStarUnit3Vec.Z() * nzJet;
              kappaDenJet = nzJet * nzJet;
              ringPerpJet = ringPrefactor * (protonLikeStarUnit3Vec.Dot(cross) * invCrossNorm - ringObservable);
              chiJet = std::atan2(-cross.Z() * v0p, cross.Y() * v0px - cross.X() * v0py);
            }
            kappaNumJet = ringObservable * ringObservable;
            // Adding prefactor
            ringObservable *= ringPrefactor;
            // Angular variables
            deltaPhiJet = wrapToPiFast(v0phi - leadingJetPhi);
            deltaEtaJet = v0eta - leadingJetEta;

            cosDeltaThetaJet = leadingJetUnitVec.Dot(lambdaLikeUnit3Vec);
            deltaThetaJet = std::acos(cosDeltaThetaJet);

            // PrimeJet-frame components, using the per-collision basis scalars:
            const float protonStarDotJet = protonLikeStarUnit3Vec.Dot(leadingJetUnitVec);
            polStarZPrimeJet = polPrefactor * protonStarDotJet; // P_{z'Jet} = p* \cdot \hat t (polarization on jet direction)
            polStarXPrimeJet = polPrefactor * (protonLikeStarUnit3Vec.Z() - jetZ * protonStarDotJet) * inverseJetTransverse; // P_{x'Jet} = (p*_z - t_z (p* \cdot \hat t)) / t_T
            polStarYPrimeJet = polPrefactor * (protonLikeStarUnit3Vec.X() * leadingJetUnitVec.Y() - protonLikeStarUnit3Vec.Y() * leadingJetUnitVec.X()) * inverseJetTransverse; // P_{y'Jet} = (p*_x t_y - p*_y t_x) / t_T

            // The same rotation is applied to the Lambda momentum to plot it in this new system:
            v0pzPrimeJet = lambdaLike3Vec.Dot(leadingJetUnitVec);
            v0pxPrimeJet = (v0pz - jetZ * v0pzPrimeJet) * inverseJetTransverse;
            v0pyPrimeJet = (v0px * leadingJetUnitVec.Y() - v0py * leadingJetUnitVec.X()) * inverseJetTransverse;

            // Testing an invariance -- <R>/\hat{t}_z -- Possible source of an artificial (trivial) sign flip in the observable:
            if (std::abs(jetZ) > 1e-4)
              ringObservableOverJetZ = ringObservable / jetZ;
            else
              ringObservableOverJetZ = 0.0; // A simple guard. May not be the best, but works

            // // A second projection schema, where e_x = normalize(t_hat cross p_Lambda), using t_hat instead of z_hat (different from phi*):
            // // (this decomposes in an orthogonal basis related to the jet coordinates)
            // XYZVector ez = lambdaLikeUnit3Vec;
            // XYZVector ex = leadingJetUnitVec.Cross(lambdaLike3Vec);
            // XYZVector ey = ez.Cross(ex);

            // ringObservableExProjection = ringObservable;
            // ringObservableEyProjection =
            // // Assuming that energy can get inside the average in:
            // // P_Lambda \cdot p_Lambda = E_Lambda/m_Lambda * P_Lambda^* \cdot p_Lambda = <E_Lambda/m_Lambda * p_D^*> * p_Lambda
            // ringObservableEzProjection = cosFakePol * lambdaLike4Vec.E()/v0LambdaLikeMass;
          }

          //////////////////////////////////////////
          // Ring observable: Subleading jet proxy
          // Only computed when a subleading jet exists in this collision.
          //////////////////////////////////////////
          float ringObservable2ndJet = 0.;
          float deltaPhi2ndJet = 0.;
          float deltaTheta2ndJet = 0.;
          float cosDeltaTheta2ndJet = 0.;
          // KappaEff moments and the R_z-only diagnostics:
          float kappaNum2ndJet = 0.;
          float kappaDen2ndJet = 1.;
          float ringPerp2ndJet = 0.;
          float chi2ndJet = 0.;
          if (hasValidSubJet) {
            XYZVector cross2ndJet = subJetUnitVec.Cross(lambdaLike3Vec);
            const float invCrossNorm2ndJet = 1.f / cross2ndJet.R();
            if (!ringZMode) {
              ringObservable2ndJet = protonLikeStarUnit3Vec.Dot(cross2ndJet) * invCrossNorm2ndJet;
            } else {
              const float nz2ndJet = cross2ndJet.Z() * invCrossNorm2ndJet;
              ringObservable2ndJet = protonLikeStarUnit3Vec.Z() * nz2ndJet;
              kappaDen2ndJet = nz2ndJet * nz2ndJet;
              ringPerp2ndJet = ringPrefactor * (protonLikeStarUnit3Vec.Dot(cross2ndJet) * invCrossNorm2ndJet - ringObservable2ndJet);
              chi2ndJet = std::atan2(-cross2ndJet.Z() * v0p, cross2ndJet.Y() * v0px - cross2ndJet.X() * v0py);
            }
            kappaNum2ndJet = ringObservable2ndJet * ringObservable2ndJet;
            // Adding prefactor
            ringObservable2ndJet *= ringPrefactor;
            // Angular variables
            deltaPhi2ndJet = wrapToPiFast(v0phi - subleadingJetPhi);
            cosDeltaTheta2ndJet = subJetUnitVec.Dot(lambdaLikeUnit3Vec);
            deltaTheta2ndJet = std::acos(cosDeltaTheta2ndJet);
          }

          float v0phiToFillHists = wrapToPiFast(v0phi); // A short wrap to reuse some predefined axes

          // R_z-only diagnostics (ringObservable* already hold R_z here):
          if (ringZMode) {
            if (hasValidLeadingJet) {
              histos.fill(HIST("RzDiagnostics/pRingVsChiLeadJet"), chiJet, ringObservable);
              histos.fill(HIST("RzDiagnostics/p2dRingVsChiVsMassLeadJet"), chiJet, v0LambdaLikeMass, ringObservable);
              histos.fill(HIST("RzDiagnostics/p2dRingVsDeltaPhiVsDeltaEtaLeadJet"), deltaPhiJet, deltaEtaJet, ringObservable);
              histos.fill(HIST("RzDiagnostics/pRingPerpIntegrated"), 0.5, ringPerpJet);
              histos.fill(HIST("RzDiagnostics/pRingPerpLeadJetVsMass"), v0LambdaLikeMass, ringPerpJet);
              histos.fill(HIST("RzDiagnostics/pRingPerpLeadJetVsPhiAEE"), deltaPhiLambdaProtonStar, ringPerpJet);
              histos.fill(HIST("RzDiagnostics/pRingPerpLeadJetVsDeltaPhi"), deltaPhiJet, ringPerpJet);
              histos.fill(HIST("RzDiagnostics/pCosSqThetaStarZLeadJetVsMass"), v0LambdaLikeMass, protonLikeStarUnit3Vec.Z() * protonLikeStarUnit3Vec.Z());
            }
            if (hasValidLeadingP) {
              histos.fill(HIST("RzDiagnostics/pRingVsChiLeadP"), chiLeadP, ringObservableLeadP);
              histos.fill(HIST("RzDiagnostics/p2dRingVsChiVsMassLeadP"), chiLeadP, v0LambdaLikeMass, ringObservableLeadP);
              histos.fill(HIST("RzDiagnostics/pRingPerpIntegrated"), 1.5, ringPerpLeadP);
              histos.fill(HIST("RzDiagnostics/pRingPerpLeadPVsMass"), v0LambdaLikeMass, ringPerpLeadP);
              histos.fill(HIST("RzDiagnostics/pRingPerpLeadPVsPhiAEE"), deltaPhiLambdaProtonStar, ringPerpLeadP);
              histos.fill(HIST("RzDiagnostics/pRingPerpLeadPVsDeltaPhi"), deltaPhiLeadP, ringPerpLeadP);
            }
            if (hasValidSubJet) {
              histos.fill(HIST("RzDiagnostics/pRingVsChiSubJet"), chi2ndJet, ringObservable2ndJet);
              histos.fill(HIST("RzDiagnostics/p2dRingVsChiVsMassSubJet"), chi2ndJet, v0LambdaLikeMass, ringObservable2ndJet);
              histos.fill(HIST("RzDiagnostics/pRingPerpIntegrated"), 2.5, ringPerp2ndJet);
              histos.fill(HIST("RzDiagnostics/pRingPerpSubJetVsMass"), v0LambdaLikeMass, ringPerp2ndJet);
              histos.fill(HIST("RzDiagnostics/pRingPerpSubJetVsPhiAEE"), deltaPhiLambdaProtonStar, ringPerp2ndJet);
              histos.fill(HIST("RzDiagnostics/pRingPerpSubJetVsDeltaPhi"), deltaPhi2ndJet, ringPerp2ndJet);
            }
          }

          // Fill ring histograms: (1D, lambda 2D correlations and jet 2D correlations):
          if (hasValidLeadingP) {
            if (familySwitches.doFamilyRing) {
              RING_OBSERVABLE_LEADP_FILL_LIST(APPLY_HISTO_FILL, "Ring") // Notice the usage of macros! If you change the variable names, this WILL break the code!
                                                                        // No, there should NOT be any ";" here! Read the macro definition for an explanation

              // Filling checks that rely on Eta>0 or Eta<0 checks for V0 and LeadingP eta:
              RING_OBSERVABLE_LEADP_ETA_SPLIT_FILL_LIST("Ring", leadPEtaPos, lambdaEtaPos);
            }
            histos.fill(HIST("IntegratedCuts/pRingCutsLeadingP"), 0, ringObservableLeadP); // First bin of comparison
            if (v0InMassPeak)
              histos.fill(HIST("IntegratedCuts/p2dRingCutsLeadingPV0MassPeak"), 0, 1, ringObservableLeadP); // Fills the inPeak bin
            else if (v0InMassWindow)
              histos.fill(HIST("IntegratedCuts/p2dRingCutsLeadingPV0MassPeak"), 0, 0, ringObservableLeadP); // Fills the inWindow bin, which has a stricter interval than !v0InMassPeak
            histos.fill(HIST("IntegratedCuts/hCountCutsLeadingP"), 0);

          }
          if (familySwitches.doFamilyRing) {
            POLARIZATION_PROFILE_FILL_LIST(APPLY_HISTO_FILL, "Ring")
          }

          // Binary search using the pre-fetched axes for delta method of error bar estimation:
          int binPt = 0; // Dummy declarations
          int binMass = 0;
          int binDTheta = 0;
          if (hasValidLeadingJet) {
            if (familySwitches.doFamilyRing) {
              RING_OBSERVABLE_FILL_LIST(APPLY_HISTO_FILL, "Ring")
            }
            histos.fill(HIST("IntegratedCuts/pRingCuts"), 0, ringObservable);
            if (v0InMassPeak)
              histos.fill(HIST("IntegratedCuts/p2dRingCutsV0MassPeak"), 0, 1, ringObservable); // Fills the inPeak bin
            else if (v0InMassWindow)
              histos.fill(HIST("IntegratedCuts/p2dRingCutsV0MassPeak"), 0, 0, ringObservable); // Fills the inWindow bin, which has a stricter interval than !v0InMassPeak
            histos.fill(HIST("IntegratedCuts/hCountCuts"), 0);
            histos.fill(HIST("IntegratedCuts/pRingVsNV0s"), nLambdaLikeV0s, ringObservable);
            histos.fill(HIST("hNV0sVsCentrality"), nLambdaLikeV0s, centrality);

            // Properly fetching values as they are needed:
            binPt = mAxisPt->FindBin(v0pt);
            binMass = mAxisMass->FindBin(v0LambdaLikeMass);
            binDTheta = mAxisDTheta->FindBin(deltaThetaJet);
            if (familySwitches.doFamilyRing) {
              trackRing.addV0(ringObservable, binPt, binMass, binDTheta);
            }

            if (qaSwitches.doFakePolDiagnosticsQA) {
              // Measuring the AEE differentially (azimuthal efficiency effect, which causes different V0 topologies to be enhanced/suppressed)
              // AEE and HEE correlation:
              histos.fill(HIST("HelicityEfficiencyQA/hFakePolCountsJet_CosThetaVsPhiStar"), cosFakePol, phiStar);
              histos.fill(HIST("HelicityEfficiencyQA/pFakePolSignal_CosThetaVsPhiStar"), cosFakePol, phiStar, ringObservable);

              // AEE and DCA between daughters correlation:
              histos.fill(HIST("HelicityEfficiencyQA/hFakePolCountsJet_PhiStarVsDCAdau"), phiStar, dcaDau);
              histos.fill(HIST("HelicityEfficiencyQA/pFakePolSignalJet_PhiStarVsDCAdau"), phiStar, dcaDau, ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/pFakePolSignalJet_PhiStarVsDCAdauVsEtaJet"), phiStar, dcaDau, leadingJetEta, ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/pFakePolSignalJet_PhiStarVsDCAdauVsEtaLambda"), phiStar, dcaDau, v0eta, ringObservable);

              // DCA dau to PV correlation:
              // For proton-like daughter:
              histos.fill(HIST("HelicityEfficiencyQA/hFakePolCountsJet_PhiStarVsDCAProLike"), phiStar, protonLikeDCADauToPV);
              histos.fill(HIST("HelicityEfficiencyQA/pFakePolSignalJet_PhiStarVsDCAProLike"), phiStar, protonLikeDCADauToPV, ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/pFakePolSignalJet_PhiStarVsDCAProLikeVsEtaJet"), phiStar, protonLikeDCADauToPV, leadingJetEta, ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/pFakePolSignalJet_PhiStarVsDCAProLikeVsEtaLambda"), phiStar, protonLikeDCADauToPV, v0eta, ringObservable);
              // For pion-like daughter:
              histos.fill(HIST("HelicityEfficiencyQA/hFakePolCountsJet_PhiStarVsDCAPiLike"), phiStar, pionLikeDCADauToPV);
              histos.fill(HIST("HelicityEfficiencyQA/pFakePolSignalJet_PhiStarVsDCAPiLike"), phiStar, pionLikeDCADauToPV, ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/pFakePolSignalJet_PhiStarVsDCAPiLikeVsEtaJet"), phiStar, pionLikeDCADauToPV, leadingJetEta, ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/pFakePolSignalJet_PhiStarVsDCAPiLikeVsEtaLambda"), phiStar, pionLikeDCADauToPV, v0eta, ringObservable);

              histos.fill(HIST("HelicityEfficiencyQA/pRingVsJetZcomponent"), jetZ, ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/pRingOverJetZcomponent_VsJetEta"), leadingJetEta, ringObservableOverJetZ);
              histos.fill(HIST("HelicityEfficiencyQA/pRingOverJetZcomponent_VsCosThetaHEE"), cosFakePol, ringObservableOverJetZ);
              histos.fill(HIST("HelicityEfficiencyQA/pRingOverJetZcomponent_VsPhiStar"), phiStar, ringObservableOverJetZ);
              histos.fill(HIST("HelicityEfficiencyQA/pRingOverJetZcomponent_VsJetEtaVsCosThetaHEE"), leadingJetEta, cosFakePol, ringObservableOverJetZ);
              histos.fill(HIST("HelicityEfficiencyQA/pRingOverJetZcomponent_VsJetEtaVsPhiStar"), leadingJetEta, phiStar, ringObservableOverJetZ);

              // HEE resolved in three mass slices, so a peak-only structure can be told apart from a flat artifact:
              const bool isAEEPhiPos = deltaPhiLambdaProtonStar >= 0;
              histos.fill(HIST("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingAllV0sCosThetaHEEVsMass"), cosFakePol, v0LambdaLikeMass, ringObservable);
              if (isAEEPhiPos)
                histos.fill(HIST("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingAllV0sCosThetaHEEVsMassPosDeltaPhiAEE"), cosFakePol, v0LambdaLikeMass, ringObservable);
              else
                histos.fill(HIST("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingAllV0sCosThetaHEEVsMassNegDeltaPhiAEE"), cosFakePol, v0LambdaLikeMass, ringObservable);
              if (isLambda) {
                histos.fill(HIST("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingLambdaCosThetaHEEVsMass"), cosFakePol, v0LambdaLikeMass, ringObservable);
                if (isAEEPhiPos)
                  histos.fill(HIST("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingLambdaCosThetaHEEVsMassPosDeltaPhiAEE"), cosFakePol, v0LambdaLikeMass, ringObservable);
                else
                  histos.fill(HIST("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingLambdaCosThetaHEEVsMassNegDeltaPhiAEE"), cosFakePol, v0LambdaLikeMass, ringObservable);
              } else {
                histos.fill(HIST("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingAntiLambdaCosThetaHEEVsMass"), cosFakePol, v0LambdaLikeMass, ringObservable);
                if (isAEEPhiPos)
                  histos.fill(HIST("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingAntiLambdaCosThetaHEEVsMassPosDeltaPhiAEE"), cosFakePol, v0LambdaLikeMass, ringObservable);
                else
                  histos.fill(HIST("HelicityEfficiencyQA/CosThetaHEEByMass/p2dRingAntiLambdaCosThetaHEEVsMassNegDeltaPhiAEE"), cosFakePol, v0LambdaLikeMass, ringObservable);
              }
            } // end doFakePolDiagnosticsQA (leading-jet AEE/HEE)
          }
          if (hasValidSubJet) {
            if (familySwitches.doFamilyRing) {
              RING_OBSERVABLE_2NDJET_FILL_LIST(APPLY_HISTO_FILL, "Ring")
            }
            histos.fill(HIST("IntegratedCuts/pRingCutsSubLeadingJet"), 0, ringObservable2ndJet);
            if (v0InMassPeak)
              histos.fill(HIST("IntegratedCuts/p2dRingCutsSubLeadingJetV0MassPeak"), 0, 1, ringObservable2ndJet); // Fills the inPeak bin
            else if (v0InMassWindow)
              histos.fill(HIST("IntegratedCuts/p2dRingCutsSubLeadingJetV0MassPeak"), 0, 0, ringObservable2ndJet); // Fills the inWindow bin, which has a stricter interval than !v0InMassPeak
            histos.fill(HIST("IntegratedCuts/hCountCutsSubLeadingJet"), 0);
          }

          if (qaSwitches.doFakePolDiagnosticsQA) {
            // Filling eta dependence QAs of the result (both for V0 and jet proxy):
            // Defining shared binning which depend on the V0 only:
            const int etaLambdaBin = lambdaEtaPos ? 3 : 4;
            if (hasValidLeadingJet) {
              histos.fill(HIST("EtaStudy/pRingEtaCuts"), 0, ringObservable);
              histos.fill(HIST("EtaStudy/pRingEtaCuts"), etaLambdaBin, ringObservable);

              // Bin indices for this proxy:
              // Bin 0: all, 1/2: proxy #eta sign, 3/4: #Lambda #eta sign, 5-8: joint,
              // 9/10: |#eta_{proxy}| >= R, 11-14: strict joint.
              const int etaProxyBin = jetEtaPos ? 1 : 2;
              const int etaProxyLambdaBin = (jetEtaPos ? 5 : 7) + (lambdaEtaPos ? 0 : 1);
              const int etaProxyStrictBin = jetEtaPos ? 9 : 10;
              const int etaProxyStrictLambdaBin = (jetEtaPos ? 11 : 13) + (lambdaEtaPos ? 0 : 1);

              histos.fill(HIST("EtaStudy/pRingEtaCuts"), etaProxyBin, ringObservable);
              histos.fill(HIST("EtaStudy/pRingEtaCuts"), etaProxyLambdaBin, ringObservable);

              // HEE study (helicity efficiency effect):
              histos.fill(HIST("EtaStudy/hFakePolCounts"), cosFakePol, 0);
              histos.fill(HIST("EtaStudy/hFakePolCounts"), cosFakePol, etaLambdaBin);
              histos.fill(HIST("EtaStudy/hFakePolCounts"), cosFakePol, etaProxyBin);
              histos.fill(HIST("EtaStudy/hFakePolCounts"), cosFakePol, etaProxyLambdaBin);
              // Same for signal:
              histos.fill(HIST("EtaStudy/pFakePolSignalVsCosTheta"), cosFakePol, 0, ringObservable);
              histos.fill(HIST("EtaStudy/pFakePolSignalVsCosTheta"), cosFakePol, etaLambdaBin, ringObservable);
              histos.fill(HIST("EtaStudy/pFakePolSignalVsCosTheta"), cosFakePol, etaProxyBin, ringObservable);
              histos.fill(HIST("EtaStudy/pFakePolSignalVsCosTheta"), cosFakePol, etaProxyLambdaBin, ringObservable);
              // Counter and ring accumulators for AEE study:
              histos.fill(HIST("EtaStudy/hCountsVsPhiStar"), phiStar, 0);
              histos.fill(HIST("EtaStudy/hCountsVsPhiStar"), phiStar, etaLambdaBin);
              histos.fill(HIST("EtaStudy/hCountsVsPhiStar"), phiStar, etaProxyBin);
              histos.fill(HIST("EtaStudy/hCountsVsPhiStar"), phiStar, etaProxyLambdaBin);
              histos.fill(HIST("EtaStudy/pFakePolSignalvsPhiStar"), phiStar, 0, ringObservable);
              histos.fill(HIST("EtaStudy/pFakePolSignalvsPhiStar"), phiStar, etaLambdaBin, ringObservable);
              histos.fill(HIST("EtaStudy/pFakePolSignalvsPhiStar"), phiStar, etaProxyBin, ringObservable);
              histos.fill(HIST("EtaStudy/pFakePolSignalvsPhiStar"), phiStar, etaProxyLambdaBin, ringObservable);

              histos.fill(HIST("EtaStudy/pFakePolSignalvsPhiLambdaMinusPhiProtonStar"), deltaPhiLambdaProtonStar, 0, ringObservable);
              histos.fill(HIST("EtaStudy/pFakePolSignalvsPhiLambdaMinusPhiProtonStar"), deltaPhiLambdaProtonStar, etaLambdaBin, ringObservable);
              histos.fill(HIST("EtaStudy/pFakePolSignalvsPhiLambdaMinusPhiProtonStar"), deltaPhiLambdaProtonStar, etaProxyBin, ringObservable);
              histos.fill(HIST("EtaStudy/pFakePolSignalvsPhiLambdaMinusPhiProtonStar"), deltaPhiLambdaProtonStar, etaProxyLambdaBin, ringObservable);

              // Inclusive and split-by-species dependencies for signal extraction:
              histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/pRingObservableLeadJetVsPhiLambdaLikePhiProtonStar"), deltaPhiLambdaProtonStar, ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiLambdaLikePhiProtonStarVsMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiLambdaLikePhiProtonStarVsProxyEta"), deltaPhiLambdaProtonStar, leadingJetEta, ringObservable);
              histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p3dRingObservableLeadJetVsPhiLambdaLikePhiProtonStarVsProxyEtaVsMass"), deltaPhiLambdaProtonStar, leadingJetEta, v0LambdaLikeMass, ringObservable);
              if (jetEtaPos) {
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiLambdaLikePhiProtonStarVsMassPosProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable);
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiLambdaLikePhiProtonStarVsProxyEtaSign"), deltaPhiLambdaProtonStar, 0.5, ringObservable);
                // Background stabilization using etaLambda differentially:
                if (deltaPhiLambdaProtonStar >= 0) {
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/BkgStabilizationAttempt/p2dRingObservableEtaLambdaVsMassEtaProxyPosPhiAEEPos"), v0eta, v0LambdaLikeMass, ringObservable);
                  if (deltaPhiJet >= 0)
                    histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/BkgStabilizationAttempt/p2dRingObservableEtaLambdaVsMassEtaProxyPosPhiAEEPosDeltaPhiJetPos"), v0eta, v0LambdaLikeMass, ringObservable);
                  else
                    histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/BkgStabilizationAttempt/p2dRingObservableEtaLambdaVsMassEtaProxyPosPhiAEEPosDeltaPhiJetNeg"), v0eta, v0LambdaLikeMass, ringObservable);
                }
                // Background stabilization attempt with etaLambda sign only, PhiAEE differentially:
                if (v0eta >= 0) {
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/BkgStabilizationAttempt/p2dRingObservablePhiLambdaPhiProtonStarVsMassEtaProxyPosPhiEtaLambdaPos"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable);
                  if (deltaPhiJet >= 0)
                    histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/BkgStabilizationAttempt/p2dRingObservablePhiLambdaPhiProtonStarVsMassEtaProxyPosPhiEtaLambdaPosDeltaPhiJetPos"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable);
                  else
                    histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/BkgStabilizationAttempt/p2dRingObservablePhiLambdaPhiProtonStarVsMassEtaProxyPosPhiEtaLambdaPosDeltaPhiJetNeg"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable);
                }
              } else {
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiLambdaLikePhiProtonStarVsMassNegProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable);
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiLambdaLikePhiProtonStarVsProxyEtaSign"), deltaPhiLambdaProtonStar, -0.5, ringObservable);
                // Background stabilization using etaLambda differentially:
                if (deltaPhiLambdaProtonStar >= 0)
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/BkgStabilizationAttempt/p2dRingObservableEtaLambdaVsMassEtaProxyNegPhiAEEPos"), v0eta, v0LambdaLikeMass, ringObservable);
                // Background stabilization attempt with etaLambda sign only, PhiAEE differentially:
                if (v0eta >= 0)
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/BkgStabilizationAttempt/p2dRingObservablePhiLambdaPhiProtonStarVsMassEtaProxyNegPhiEtaLambdaPos"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable);
              }
              histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/ThreeBinMass/p2dRingObservableLeadJetVsPhiLambdaLikePhiProtonStarVs3BinMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable);
              if (isLambda) {
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/pRingObservableLeadJetVsPhiLambdaPhiProtonStar"), deltaPhiLambdaProtonStar, ringObservable);
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiLambdaPhiProtonStarVsMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable);
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiLambdaPhiProtonStarVsProxyEta"), deltaPhiLambdaProtonStar, leadingJetEta, ringObservable);
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p3dRingObservableLeadJetVsPhiLambdaPhiProtonStarVsProxyEtaVsMass"), deltaPhiLambdaProtonStar, leadingJetEta, v0LambdaLikeMass, ringObservable);
                if (jetEtaPos) {
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiLambdaPhiProtonStarVsMassPosProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable);
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiLambdaPhiProtonStarVsProxyEtaSign"), deltaPhiLambdaProtonStar, 0.5, ringObservable);
                } else {
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiLambdaPhiProtonStarVsMassNegProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable);
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiLambdaPhiProtonStarVsProxyEtaSign"), deltaPhiLambdaProtonStar, -0.5, ringObservable);
                }
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/ThreeBinMass/p2dRingObservableLeadJetVsPhiLambdaPhiProtonStarVs3BinMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable);
              } else {
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/pRingObservableLeadJetVsPhiAntiLambdaPhiProtonStar"), deltaPhiLambdaProtonStar, ringObservable);
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiAntiLambdaPhiProtonStarVsMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable);
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiAntiLambdaPhiProtonStarVsProxyEta"), deltaPhiLambdaProtonStar, leadingJetEta, ringObservable);
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p3dRingObservableLeadJetVsPhiAntiLambdaPhiProtonStarVsProxyEtaVsMass"), deltaPhiLambdaProtonStar, leadingJetEta, v0LambdaLikeMass, ringObservable);
                if (jetEtaPos) {
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiAntiLambdaPhiProtonStarVsMassPosProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable);
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiAntiLambdaPhiProtonStarVsProxyEtaSign"), deltaPhiLambdaProtonStar, 0.5, ringObservable);
                } else {
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiAntiLambdaPhiProtonStarVsMassNegProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable);
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadJetVsPhiAntiLambdaPhiProtonStarVsProxyEtaSign"), deltaPhiLambdaProtonStar, -0.5, ringObservable);
                }
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/ThreeBinMass/p2dRingObservableLeadJetVsPhiAntiLambdaPhiProtonStarVs3BinMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable);
              }

              // Extra correlations test:
              histos.fill(HIST("HelicityEfficiencyQA/hFakePolCountsVsDeltaThetaJet"), cosFakePol, deltaThetaJet);
              histos.fill(HIST("HelicityEfficiencyQA/hFakePolCountsCosThetaVsPtForJets"), cosFakePol, v0pt);
              // Split by proxy #eta sign:
              if (jetEtaPos)
                histos.fill(HIST("HelicityEfficiencyQA/hFakePolCountsVsDeltaThetaJetPosEta"), cosFakePol, deltaThetaJet);
              else
                histos.fill(HIST("HelicityEfficiencyQA/hFakePolCountsVsDeltaThetaJetNegEta"), cosFakePol, deltaThetaJet);

              if (pTLambdaCheck) {
                histos.fill(HIST("EtaStudy/hFakePolCountsLambdaPtCut"), cosFakePol, 0);
                histos.fill(HIST("EtaStudy/hFakePolCountsLambdaPtCut"), cosFakePol, etaLambdaBin);
                histos.fill(HIST("EtaStudy/hFakePolCountsLambdaPtCut"), cosFakePol, etaProxyBin);
                histos.fill(HIST("EtaStudy/hFakePolCountsLambdaPtCut"), cosFakePol, etaProxyLambdaBin);
                if (rapidityLambdaCheck) { // Stricter check
                  histos.fill(HIST("EtaStudy/hFakePolCountsLambdaPtYCuts"), cosFakePol, 0);
                  histos.fill(HIST("EtaStudy/hFakePolCountsLambdaPtYCuts"), cosFakePol, etaLambdaBin);
                  histos.fill(HIST("EtaStudy/hFakePolCountsLambdaPtYCuts"), cosFakePol, etaProxyBin);
                  histos.fill(HIST("EtaStudy/hFakePolCountsLambdaPtYCuts"), cosFakePol, etaProxyLambdaBin);
                }
              }
              if (jetEtaStrict) { // |eta_{Jet}| >= R
                histos.fill(HIST("EtaStudy/pRingEtaCuts"), etaProxyStrictBin, ringObservable);
                histos.fill(HIST("EtaStudy/pRingEtaCuts"), etaProxyStrictLambdaBin, ringObservable);
              }
            }
            if (hasValidSubJet) {
              // Same bin scheme as the leading jet above:
              const int etaProxyBin = subJetEtaPos ? 1 : 2;
              const int etaProxyLambdaBin = (subJetEtaPos ? 5 : 7) + (lambdaEtaPos ? 0 : 1);
              const int etaProxyStrictBin = subJetEtaPos ? 9 : 10;
              const int etaProxyStrictLambdaBin = (subJetEtaPos ? 11 : 13) + (lambdaEtaPos ? 0 : 1);

              histos.fill(HIST("EtaStudy/pRingEtaCutsSubLeadingJet"), 0, ringObservable2ndJet);
              histos.fill(HIST("EtaStudy/pRingEtaCutsSubLeadingJet"), etaLambdaBin, ringObservable2ndJet);
              histos.fill(HIST("EtaStudy/pRingEtaCutsSubLeadingJet"), etaProxyBin, ringObservable2ndJet);
              histos.fill(HIST("EtaStudy/pRingEtaCutsSubLeadingJet"), etaProxyLambdaBin, ringObservable2ndJet);
              if (subJetEtaStrict) { // |eta_{SubJet}| >= R
                histos.fill(HIST("EtaStudy/pRingEtaCutsSubLeadingJet"), etaProxyStrictBin, ringObservable2ndJet);
                histos.fill(HIST("EtaStudy/pRingEtaCutsSubLeadingJet"), etaProxyStrictLambdaBin, ringObservable2ndJet);
              }

              // Inclusive and split-by-species dependencies for signal extraction (AEE):
              histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/pRingObservable2ndJetVsPhiLambdaLikePhiProtonStar"), deltaPhiLambdaProtonStar, ringObservable2ndJet);
              histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiLambdaLikePhiProtonStarVsMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable2ndJet);
              histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiLambdaLikePhiProtonStarVsProxyEta"), deltaPhiLambdaProtonStar, subleadingJetEta, ringObservable2ndJet);
              histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p3dRingObservable2ndJetVsPhiLambdaLikePhiProtonStarVsProxyEtaVsMass"), deltaPhiLambdaProtonStar, subleadingJetEta, v0LambdaLikeMass, ringObservable2ndJet);
              if (subJetEtaPos) {
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiLambdaLikePhiProtonStarVsMassPosProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable2ndJet);
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiLambdaLikePhiProtonStarVsProxyEtaSign"), deltaPhiLambdaProtonStar, 0.5, ringObservable2ndJet);
              } else {
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiLambdaLikePhiProtonStarVsMassNegProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable2ndJet);
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiLambdaLikePhiProtonStarVsProxyEtaSign"), deltaPhiLambdaProtonStar, -0.5, ringObservable2ndJet);
              }
              histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/ThreeBinMass/p2dRingObservable2ndJetVsPhiLambdaLikePhiProtonStarVs3BinMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable2ndJet);
              if (isLambda) {
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/pRingObservable2ndJetVsPhiLambdaPhiProtonStar"), deltaPhiLambdaProtonStar, ringObservable2ndJet);
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiLambdaPhiProtonStarVsMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable2ndJet);
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiLambdaPhiProtonStarVsProxyEta"), deltaPhiLambdaProtonStar, subleadingJetEta, ringObservable2ndJet);
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p3dRingObservable2ndJetVsPhiLambdaPhiProtonStarVsProxyEtaVsMass"), deltaPhiLambdaProtonStar, subleadingJetEta, v0LambdaLikeMass, ringObservable2ndJet);
                if (subJetEtaPos) {
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiLambdaPhiProtonStarVsMassPosProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable2ndJet);
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiLambdaPhiProtonStarVsProxyEtaSign"), deltaPhiLambdaProtonStar, 0.5, ringObservable2ndJet);
                } else {
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiLambdaPhiProtonStarVsMassNegProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable2ndJet);
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiLambdaPhiProtonStarVsProxyEtaSign"), deltaPhiLambdaProtonStar, -0.5, ringObservable2ndJet);
                }
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/ThreeBinMass/p2dRingObservable2ndJetVsPhiLambdaPhiProtonStarVs3BinMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable2ndJet);
              } else {
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/pRingObservable2ndJetVsPhiAntiLambdaPhiProtonStar"), deltaPhiLambdaProtonStar, ringObservable2ndJet);
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiAntiLambdaPhiProtonStarVsMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable2ndJet);
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiAntiLambdaPhiProtonStarVsProxyEta"), deltaPhiLambdaProtonStar, subleadingJetEta, ringObservable2ndJet);
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p3dRingObservable2ndJetVsPhiAntiLambdaPhiProtonStarVsProxyEtaVsMass"), deltaPhiLambdaProtonStar, subleadingJetEta, v0LambdaLikeMass, ringObservable2ndJet);
                if (subJetEtaPos) {
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiAntiLambdaPhiProtonStarVsMassPosProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable2ndJet);
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiAntiLambdaPhiProtonStarVsProxyEtaSign"), deltaPhiLambdaProtonStar, 0.5, ringObservable2ndJet);
                } else {
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiAntiLambdaPhiProtonStarVsMassNegProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable2ndJet);
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservable2ndJetVsPhiAntiLambdaPhiProtonStarVsProxyEtaSign"), deltaPhiLambdaProtonStar, -0.5, ringObservable2ndJet);
                }
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/ThreeBinMass/p2dRingObservable2ndJetVsPhiAntiLambdaPhiProtonStarVs3BinMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservable2ndJet);
              }
            }
            if (hasValidLeadingP) {
              // Same bin scheme, without the strict variant (this axis stops at 9 bins):
              const int etaProxyBin = leadPEtaPos ? 1 : 2;
              const int etaProxyLambdaBin = (leadPEtaPos ? 5 : 7) + (lambdaEtaPos ? 0 : 1);

              histos.fill(HIST("EtaStudy/pRingEtaCutsLeadingP"), 0, ringObservableLeadP);
              histos.fill(HIST("EtaStudy/pRingEtaCutsLeadingP"), etaLambdaBin, ringObservableLeadP);
              histos.fill(HIST("EtaStudy/pRingEtaCutsLeadingP"), etaProxyBin, ringObservableLeadP);
              histos.fill(HIST("EtaStudy/pRingEtaCutsLeadingP"), etaProxyLambdaBin, ringObservableLeadP);
              if (v0InMassPeak) {
                histos.fill(HIST("EtaStudy/pRingEtaCutsLeadingP_MassSignalVsBackground"), 0, 1, ringObservableLeadP);
                histos.fill(HIST("EtaStudy/pRingEtaCutsLeadingP_MassSignalVsBackground"), etaLambdaBin, 1, ringObservableLeadP);
                histos.fill(HIST("EtaStudy/pRingEtaCutsLeadingP_MassSignalVsBackground"), etaProxyBin, 1, ringObservableLeadP);
                histos.fill(HIST("EtaStudy/pRingEtaCutsLeadingP_MassSignalVsBackground"), etaProxyLambdaBin, 1, ringObservableLeadP);
              } else if (v0InMassWindow) {
                histos.fill(HIST("EtaStudy/pRingEtaCutsLeadingP_MassSignalVsBackground"), 0, 0, ringObservableLeadP);
                histos.fill(HIST("EtaStudy/pRingEtaCutsLeadingP_MassSignalVsBackground"), etaLambdaBin, 0, ringObservableLeadP);
                histos.fill(HIST("EtaStudy/pRingEtaCutsLeadingP_MassSignalVsBackground"), etaProxyBin, 0, ringObservableLeadP);
                histos.fill(HIST("EtaStudy/pRingEtaCutsLeadingP_MassSignalVsBackground"), etaProxyLambdaBin, 0, ringObservableLeadP);
              }
              histos.fill(HIST("EtaStudy/hFakePolCountsLeadP"), cosFakePol, 0);
              histos.fill(HIST("EtaStudy/hFakePolCountsLeadP"), cosFakePol, etaLambdaBin);
              histos.fill(HIST("EtaStudy/hFakePolCountsLeadP"), cosFakePol, etaProxyBin);
              histos.fill(HIST("EtaStudy/hFakePolCountsLeadP"), cosFakePol, etaProxyLambdaBin);

              histos.fill(HIST("HelicityEfficiencyQA/hFakePolCountsCosThetaVsPtForLeadP"), cosFakePol, v0pt); // Understanding the population of events that has a leading particle (even though this does not need one to be calculated!)
              histos.fill(HIST("HelicityEfficiencyQA/hFakePolCountsVsDeltaThetaLeadP"), cosFakePol, deltaThetaLeadP);
              if (leadPEtaPos)
                histos.fill(HIST("HelicityEfficiencyQA/hFakePolCountsVsDeltaThetaLeadPPosEta"), cosFakePol, deltaThetaLeadP);
              else
                histos.fill(HIST("HelicityEfficiencyQA/hFakePolCountsVsDeltaThetaLeadPNegEta"), cosFakePol, deltaThetaLeadP);

              // Inclusive and split-by-species dependencies for signal extraction (AEE):
              histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/pRingObservableLeadPVsPhiLambdaLikePhiProtonStar"), deltaPhiLambdaProtonStar, ringObservableLeadP);
              histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiLambdaLikePhiProtonStarVsMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservableLeadP);
              histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiLambdaLikePhiProtonStarVsProxyEta"), deltaPhiLambdaProtonStar, leadPEta, ringObservableLeadP);
              histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p3dRingObservableLeadPVsPhiLambdaLikePhiProtonStarVsProxyEtaVsMass"), deltaPhiLambdaProtonStar, leadPEta, v0LambdaLikeMass, ringObservableLeadP);
              if (leadPEtaPos) {
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiLambdaLikePhiProtonStarVsMassPosProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservableLeadP);
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiLambdaLikePhiProtonStarVsProxyEtaSign"), deltaPhiLambdaProtonStar, 0.5, ringObservableLeadP);
              } else {
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiLambdaLikePhiProtonStarVsMassNegProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservableLeadP);
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiLambdaLikePhiProtonStarVsProxyEtaSign"), deltaPhiLambdaProtonStar, -0.5, ringObservableLeadP);
              }
              histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/ThreeBinMass/p2dRingObservableLeadPVsPhiLambdaLikePhiProtonStarVs3BinMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservableLeadP);
              if (isLambda) {
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/pRingObservableLeadPVsPhiLambdaPhiProtonStar"), deltaPhiLambdaProtonStar, ringObservableLeadP);
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiLambdaPhiProtonStarVsMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservableLeadP);
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiLambdaPhiProtonStarVsProxyEta"), deltaPhiLambdaProtonStar, leadPEta, ringObservableLeadP);
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p3dRingObservableLeadPVsPhiLambdaPhiProtonStarVsProxyEtaVsMass"), deltaPhiLambdaProtonStar, leadPEta, v0LambdaLikeMass, ringObservableLeadP);
                if (leadPEtaPos) {
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiLambdaPhiProtonStarVsMassPosProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservableLeadP);
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiLambdaPhiProtonStarVsProxyEtaSign"), deltaPhiLambdaProtonStar, 0.5, ringObservableLeadP);
                } else {
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiLambdaPhiProtonStarVsMassNegProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservableLeadP);
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiLambdaPhiProtonStarVsProxyEtaSign"), deltaPhiLambdaProtonStar, -0.5, ringObservableLeadP);
                }
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/ThreeBinMass/p2dRingObservableLeadPVsPhiLambdaPhiProtonStarVs3BinMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservableLeadP);
              } else {
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/pRingObservableLeadPVsPhiAntiLambdaPhiProtonStar"), deltaPhiLambdaProtonStar, ringObservableLeadP);
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiAntiLambdaPhiProtonStarVsMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservableLeadP);
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiAntiLambdaPhiProtonStarVsProxyEta"), deltaPhiLambdaProtonStar, leadPEta, ringObservableLeadP);
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p3dRingObservableLeadPVsPhiAntiLambdaPhiProtonStarVsProxyEtaVsMass"), deltaPhiLambdaProtonStar, leadPEta, v0LambdaLikeMass, ringObservableLeadP);
                if (leadPEtaPos) {
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiAntiLambdaPhiProtonStarVsMassPosProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservableLeadP);
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiAntiLambdaPhiProtonStarVsProxyEtaSign"), deltaPhiLambdaProtonStar, 0.5, ringObservableLeadP);
                } else {
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiAntiLambdaPhiProtonStarVsMassNegProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservableLeadP);
                  histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/p2dRingObservableLeadPVsPhiAntiLambdaPhiProtonStarVsProxyEtaSign"), deltaPhiLambdaProtonStar, -0.5, ringObservableLeadP);
                }
                histos.fill(HIST("HelicityEfficiencyQA/PhiLambdaPhiProtonStar/ThreeBinMass/p2dRingObservableLeadPVsPhiAntiLambdaPhiProtonStarVs3BinMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringObservableLeadP);
              }
            }
          } // end doFakePolDiagnosticsQA (eta-dependence block)

          // Extra kinematic criteria for Lambda candidates (removes polarization background):
          if (kinematicLambdaCheck) {
            if (hasValidLeadingP) {
              if (familySwitches.doFamilyRingKinematicCuts) {
                RING_OBSERVABLE_LEADP_FILL_LIST(APPLY_HISTO_FILL, "RingKinematicCuts")

                // Filling checks that rely on Eta>0 or Eta<0 checks for V0 and LeadingP eta:
                RING_OBSERVABLE_LEADP_ETA_SPLIT_FILL_LIST("RingKinematicCuts", leadPEtaPos, lambdaEtaPos);
              }
              histos.fill(HIST("IntegratedCuts/pRingCutsLeadingP"), 1, ringObservableLeadP);
              if (v0InMassPeak)
                histos.fill(HIST("IntegratedCuts/p2dRingCutsLeadingPV0MassPeak"), 1, 1, ringObservableLeadP); // Fills the inPeak bin
              else if (v0InMassWindow)
                histos.fill(HIST("IntegratedCuts/p2dRingCutsLeadingPV0MassPeak"), 1, 0, ringObservableLeadP); // Fills the inWindow bin, which has a stricter interval than !v0InMassPeak
              histos.fill(HIST("IntegratedCuts/hCountCutsLeadingP"), 1);
            }
            if (familySwitches.doFamilyRingKinematicCuts) {
              POLARIZATION_PROFILE_FILL_LIST(APPLY_HISTO_FILL, "RingKinematicCuts")
            }
            if (hasValidLeadingJet) {
              if (familySwitches.doFamilyRingKinematicCuts) {
                RING_OBSERVABLE_FILL_LIST(APPLY_HISTO_FILL, "RingKinematicCuts")
              }
              histos.fill(HIST("IntegratedCuts/pRingCuts"), 1, ringObservable);
              if (v0InMassPeak)
                histos.fill(HIST("IntegratedCuts/p2dRingCutsV0MassPeak"), 1, 1, ringObservable); // Fills the inPeak bin
              else if (v0InMassWindow)
                histos.fill(HIST("IntegratedCuts/p2dRingCutsV0MassPeak"), 1, 0, ringObservable); // Fills the inWindow bin, which has a stricter interval than !v0InMassPeak
              histos.fill(HIST("IntegratedCuts/hCountCuts"), 1);
              if (familySwitches.doFamilyRingKinematicCuts) {
                trackRingKinCuts.addV0(ringObservable, binPt, binMass, binDTheta);
              }
            }
            if (hasValidSubJet) {
              if (familySwitches.doFamilyRingKinematicCuts) {
                RING_OBSERVABLE_2NDJET_FILL_LIST(APPLY_HISTO_FILL, "RingKinematicCuts")
              }
              histos.fill(HIST("IntegratedCuts/pRingCutsSubLeadingJet"), 1, ringObservable2ndJet);
              if (v0InMassPeak)
                histos.fill(HIST("IntegratedCuts/p2dRingCutsSubLeadingJetV0MassPeak"), 1, 1, ringObservable2ndJet); // Fills the inPeak bin
              else if (v0InMassWindow)
                histos.fill(HIST("IntegratedCuts/p2dRingCutsSubLeadingJetV0MassPeak"), 1, 0, ringObservable2ndJet); // Fills the inWindow bin, which has a stricter interval than !v0InMassPeak
              histos.fill(HIST("IntegratedCuts/hCountCutsSubLeadingJet"), 1);
            }
          }

          // Extra selection criteria on jet candidates:
          // (redundant for jets with R=0.4, but for jets with R<0.4 the leading jet may be farther in eta)
          if (kinematicJetCheck) { // Already includes hasValidLeadingJet in the bool! (no need to check again)
            if (familySwitches.doFamilyJetKinematicCuts) {
              RING_OBSERVABLE_FILL_LIST(APPLY_HISTO_FILL, "JetKinematicCuts")
            }
            histos.fill(HIST("IntegratedCuts/pRingCuts"), 2, ringObservable);
            if (v0InMassPeak)
              histos.fill(HIST("IntegratedCuts/p2dRingCutsV0MassPeak"), 2, 1, ringObservable); // Fills the inPeak bin
            else if (v0InMassWindow)
              histos.fill(HIST("IntegratedCuts/p2dRingCutsV0MassPeak"), 2, 0, ringObservable); // Fills the inWindow bin, which has a stricter interval than !v0InMassPeak
            histos.fill(HIST("IntegratedCuts/hCountCuts"), 2);
            if (familySwitches.doFamilyJetKinematicCuts) {
              POLARIZATION_PROFILE_FILL_LIST(APPLY_HISTO_FILL, "JetKinematicCuts")
              trackJetKinCuts.addV0(ringObservable, binPt, binMass, binDTheta);
            }
          }

          // Extra selection criteria on both Lambda and jet candidates:
          if (kinematicLambdaCheck && kinematicJetCheck) {
            if (familySwitches.doFamilyJetAndLambdaKinematicCuts) {
              RING_OBSERVABLE_FILL_LIST(APPLY_HISTO_FILL, "JetAndLambdaKinematicCuts")
            }
            histos.fill(HIST("IntegratedCuts/pRingCuts"), 3, ringObservable);
            if (v0InMassPeak)
              histos.fill(HIST("IntegratedCuts/p2dRingCutsV0MassPeak"), 3, 1, ringObservable); // Fills the inPeak bin
            else if (v0InMassWindow)
              histos.fill(HIST("IntegratedCuts/p2dRingCutsV0MassPeak"), 3, 0, ringObservable); // Fills the inWindow bin, which has a stricter interval than !v0InMassPeak
            histos.fill(HIST("IntegratedCuts/hCountCuts"), 3);
            if (familySwitches.doFamilyJetAndLambdaKinematicCuts) {
              POLARIZATION_PROFILE_FILL_LIST(APPLY_HISTO_FILL, "JetAndLambdaKinematicCuts")
              trackJetLambdaKinCuts.addV0(ringObservable, binPt, binMass, binDTheta);
            }
          }

          // Same variations for the leading particle and for the subleading jet:
          // (kinematicLeadPCheck already encodes hasValidLeadingP, so no extra gate needed here)
          if (kinematicLeadPCheck) {
            if (familySwitches.doFamilyJetKinematicCuts) {
              RING_OBSERVABLE_LEADP_FILL_LIST(APPLY_HISTO_FILL, "JetKinematicCuts")

              // Filling checks that rely on Eta>0 or Eta<0 checks for V0 and LeadingP eta:
              RING_OBSERVABLE_LEADP_ETA_SPLIT_FILL_LIST("JetKinematicCuts", leadPEtaPos, lambdaEtaPos);
            }
            histos.fill(HIST("IntegratedCuts/pRingCutsLeadingP"), 2, ringObservableLeadP);
            if (v0InMassPeak)
              histos.fill(HIST("IntegratedCuts/p2dRingCutsLeadingPV0MassPeak"), 2, 1, ringObservableLeadP); // Fills the inPeak bin
            else if (v0InMassWindow)
              histos.fill(HIST("IntegratedCuts/p2dRingCutsLeadingPV0MassPeak"), 2, 0, ringObservableLeadP); // Fills the inWindow bin, which has a stricter interval than !v0InMassPeak
            histos.fill(HIST("IntegratedCuts/hCountCutsLeadingP"), 2);
          }
          if (kinematic2ndJetCheck) {
            if (familySwitches.doFamilyJetKinematicCuts) {
              RING_OBSERVABLE_2NDJET_FILL_LIST(APPLY_HISTO_FILL, "JetKinematicCuts")
            }
            histos.fill(HIST("IntegratedCuts/pRingCutsSubLeadingJet"), 2, ringObservable2ndJet);
            if (v0InMassPeak)
              histos.fill(HIST("IntegratedCuts/p2dRingCutsSubLeadingJetV0MassPeak"), 2, 1, ringObservable2ndJet); // Fills the inPeak bin
            else if (v0InMassWindow)
              histos.fill(HIST("IntegratedCuts/p2dRingCutsSubLeadingJetV0MassPeak"), 2, 0, ringObservable2ndJet); // Fills the inWindow bin, which has a stricter interval than !v0InMassPeak
            histos.fill(HIST("IntegratedCuts/hCountCutsSubLeadingJet"), 2);
          }
          if (kinematicLambdaCheck && kinematicLeadPCheck) {
            if (familySwitches.doFamilyJetAndLambdaKinematicCuts) {
              RING_OBSERVABLE_LEADP_FILL_LIST(APPLY_HISTO_FILL, "JetAndLambdaKinematicCuts")

              // Filling checks that rely on Eta>0 or Eta<0 checks for V0 and LeadingP eta:
              RING_OBSERVABLE_LEADP_ETA_SPLIT_FILL_LIST("JetAndLambdaKinematicCuts", leadPEtaPos, lambdaEtaPos);
            }
            histos.fill(HIST("IntegratedCuts/pRingCutsLeadingP"), 3, ringObservableLeadP);
            if (v0InMassPeak)
              histos.fill(HIST("IntegratedCuts/p2dRingCutsLeadingPV0MassPeak"), 3, 1, ringObservableLeadP); // Fills the inPeak bin
            else if (v0InMassWindow)
              histos.fill(HIST("IntegratedCuts/p2dRingCutsLeadingPV0MassPeak"), 3, 0, ringObservableLeadP); // Fills the inWindow bin, which has a stricter interval than !v0InMassPeak
            histos.fill(HIST("IntegratedCuts/hCountCutsLeadingP"), 3);
          }
          if (kinematicLambdaCheck && kinematic2ndJetCheck) {
            if (familySwitches.doFamilyJetAndLambdaKinematicCuts) {
              RING_OBSERVABLE_2NDJET_FILL_LIST(APPLY_HISTO_FILL, "JetAndLambdaKinematicCuts")
            }
            histos.fill(HIST("IntegratedCuts/pRingCutsSubLeadingJet"), 3, ringObservable2ndJet);
            if (v0InMassPeak)
              histos.fill(HIST("IntegratedCuts/p2dRingCutsSubLeadingJetV0MassPeak"), 3, 1, ringObservable2ndJet); // Fills the inPeak bin
            else if (v0InMassWindow)
              histos.fill(HIST("IntegratedCuts/p2dRingCutsSubLeadingJetV0MassPeak"), 3, 0, ringObservable2ndJet); // Fills the inWindow bin, which has a stricter interval than !v0InMassPeak
            histos.fill(HIST("IntegratedCuts/hCountCutsSubLeadingJet"), 3);
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
