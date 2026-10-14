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

#include <algorithm> // std::fill, for resetting the Delta Method accumulators. std::upper_bound, for the mass regions
#include <array>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <type_traits> // std::integral_constant, for the ProxyTypeTag of the fill lambdas
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

// Defining an enum for the different jet proxy types:
// (this will be used to get some compile-time indices via integral_constant below, for using HIST()'s compile-time capabilities)
// (the values also double as bin indices for the RzDiagnostics/*Integrated profiles)
enum ProxyTypeIndex {
  kLeadJet = 0,
  kLeadP,
  kSubJet,
  kNProxyTypes
};

// Folder names, indexed by the enum above (HIST() accepts a constexpr string_view, and HIST() names concatenate with "+"):
// (last entry also corresponds to the total number of entries in the array, so we use that to avoid magic numbers)
constexpr std::array<std::string_view, kNProxyTypes> ProxyNames = {"LeadJet", "LeadP", "SubJet"};

// Transforming the enum into actual compile-time tags for the fill lambdas in the main loop:
// (a lambda receiving these reads the index with decltype(tag)::value)
template <int idxProxy>
constexpr std::integral_constant<int, idxProxy> ProxyTypeTag{};

// CheapSigExtract mass regions, built in init():
// (edges at LambdaMass -/+ SidebandOuterNSigma, SidebandInnerNSigma*LambdaMassSigma and PeakWindowNSigma*LambdaMassSigma)
enum MassRegion {
  kOutsideLow = 0, // underflow
  kLeftSideband,
  kLeftGap,
  kPeak,
  kRightGap,
  kRightSideband,
  kOutsideHigh // overflow
};

struct lambdajetpolarizationionsderived {
  // Define histogram registries:
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};
  HistogramRegistry histosRing{"HistosRing", {}, OutputObjHandlingPolicy::AnalysisObject};

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

  // QA switches:
  struct : ConfigurableGroup {
    std::string prefix = "qaSwitches";                                                                                      // JSON group name
    Configurable<bool> doFakePolDiagnosticsQA{"doFakePolDiagnosticsQA", true, "Book and fill the FakePolDrivers/ folder."}; // The largest per-V0 fill cost in this task
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
    Configurable<float> maxJetProxyEta{"maxJetProxyEta", 0.5f, "|eta| ceiling for artificial leading and subleading jet proxies."};
    Configurable<int> nProxyResamples{"nProxyResamples", 1, "The amount of resamplings of jet direction per event. Use ONLY for forceRandJet, forceDatalikeJet and doMixedEventProxies"};
  } fakePolSwitches;

  Configurable<float> jetR{"jetR", 0.4f, "Radius of the jet"}; // Provide manually, please. Also used for forceJetDirectionSmudge

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
    // An additional V0-level cut on lab phi (currently for testing only):
    Configurable<std::vector<float>> v0PhiLimits{"v0PhiLimits", {0.f, constants::math::TwoPI}, "Limits on the V0 phi, in [0,2pi]. Testing only."};

    // V0 Mass cuts to remove candidates outside of the usual signal extraction limits (a cheap QA procedure before full signal extraction):
    Configurable<float> v0MassMin{"v0MassMin", 1.09f, "V0 mass min cut. Removes V0s from combinatoric background, for QA"}; // Inert cut is -999.f
    Configurable<float> v0MassMax{"v0MassMax", 1.14f, "V0 mass max cut. Removes V0s from combinatoric background, for QA"}; // Inert cut is +999.f
  } analysisLevelCuts;

  // Jet-proxy selections (minimum pT and |eta| acceptance), applied in the pre-pass, before any V0 processing (see isProxyAccepted):
  // (the |eta| defaults match the FastJet outputs and the table producer, so they only matter for stricter cuts or for jets with R < 0.4)
  struct : ConfigurableGroup {
    std::string prefix = "analysisLevelCutsJets"; // JSON group name
    Configurable<bool> doAnalysisLevelCutsJets{"doAnalysisLevelCutsJets", false, "Apply the |eta| acceptance cuts below to the jet proxies (leading jet, subleading jet and leading particle). The minimum pT is always applied."};
    // Kinematics:
    // (the minimum pT defines a valid proxy, so it is applied regardless of doAnalysisLevelCutsJets)
    Configurable<float> minLeadJetPt{"minLeadJetPt", 10.0f, "Minimum Pt for leading jet to be considered valid (may be more restrictive than TableProducer)"};
    Configurable<float> minSubJetPt{"minSubJetPt", 8.0f, "Minimum Pt for subleading jet to be considered valid (may be more restrictive than TableProducer)"};
    Configurable<float> minLeadPPt{"minLeadPPt", 4.0f, "Minimum Pt for a lead track to be considered a valid proxy for a jet (may be more restrictive than TableProducer)"};
    Configurable<float> maxLeadJetEta{"maxLeadJetEta", 0.5f, "|eta| ceiling for the leading jet. Default is the FastJet output's (0.9 - R for R = 0.4)."};
    Configurable<float> maxSubJetEta{"maxSubJetEta", 0.5f, "|eta| ceiling for the subleading jet. Default is the FastJet output's (0.9 - R for R = 0.4)."};
    Configurable<float> maxLeadPEta{"maxLeadPEta", 0.9f, "|eta| ceiling for the leading particle. Same default as the table producer."};

    // Quenching, etc.:
  } analysisLevelCutsJets;

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

  // Declaring the tracker here, so its accumulators are allocated only once (reset per collision):
  EventDeltaTracker deltaTracker;

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

  // QA guards, set once in init and shared by the bookings and the fills:
  bool doMixingQA = false;         // doMixedEventProxies && doEventMixingQA
  bool doResamplingPassQA = false; // Bool for nProxyResamples > 1

  // CheapSigExtract mass-region edges, based on the NSigma configurables, and calculated with a small helper:
  std::array<double, 6> massRegionEdges{};

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

  /// \brief Small helper for the CheapSigExtract region of a Lambda-like mass. Follows the bin convention of a regular TAxis::FindBin on the five-bin mass axis:
  ///        lower edges inclusive, upper edges exclusive, kOutsideLow below the first edge and kOutsideHigh from the last one on.
  MassRegion massRegionOf(float mass) const
  {
    return static_cast<MassRegion>(std::upper_bound(massRegionEdges.begin(), massRegionEdges.end(), static_cast<double>(mass)) - massRegionEdges.begin());
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
    if (std::abs((SidebandOuterNSigma - SidebandInnerNSigma) - PeakWindowNSigma) >= 1e-9)
      LOG(fatal) << "Sideband and peak windows must have the same width for the histograms here.";

    // CheapSigExtract mass regions (left sideband, left gap, peak, right gap, right sideband), with the NSigma configurables as the single source of truth:
    massRegionEdges = {LambdaMass - SidebandOuterNSigma * LambdaMassSigma, LambdaMass - SidebandInnerNSigma * LambdaMassSigma,
                       LambdaMass - PeakWindowNSigma * LambdaMassSigma, LambdaMass + PeakWindowNSigma * LambdaMassSigma,
                       LambdaMass + SidebandInnerNSigma * LambdaMassSigma, LambdaMass + SidebandOuterNSigma * LambdaMassSigma};
    for (size_t edge = 1; edge < massRegionEdges.size(); ++edge) {
      if (!(massRegionEdges[edge] > massRegionEdges[edge - 1]))
        LOG(fatal) << "Mass regions: the edges must strictly increase, so 0 < PeakWindowNSigma < SidebandInnerNSigma < SidebandOuterNSigma is required.";
    }
    // The same edges as a five-bin axis (masses outside the sidebands go to the under/overflow):
    const AxisSpec axisMassRegions{std::vector<double>(massRegionEdges.begin(), massRegionEdges.end())};

    if (useRingZ && useRingZT) {
      LOG(fatal) << "useRingZ and useRingZT are mutually exclusive ring definitions: enable at most one.";
    }
    longitudinalRingMode = useRingZ || useRingZT;
    transverseBeamAxisMode = useRingZT;
    // Calculating the QA guards we've defined before init():
    doMixingQA = fakePolSwitches.doMixedEventProxies && qaSwitches.doEventMixingQA;
    doResamplingPassQA = fakePolSwitches.nProxyResamples > 1;
    // forceJetDirectionSmudge width, from the configured jet radius:
    smearSigma = 0.05 * jetR;

    // V0 phi selection:
    v0PhiLimitsVec = (std::vector<float>)analysisLevelCuts.v0PhiLimits;

    // Ring observable histograms:
    // (one folder per proxy, Ring/<Proxy>/<Subfolder>/)
    // ===============================
    // Integrated observable (proxy root)
    // ===============================
    histosRing.add("Ring/LeadJet/pRingObservableIntegrated", "Integrated <#it{R}>; ;<#it{R}>", kTProfile, {{1, -0.5, 0.5}});
    histosRing.add("Ring/LeadJet/hIntegrated", "Integrated counts; ;Counts", kTH1D, {{1, -0.5, 0.5}});
    // To see the actual distribution of counts in data (another differential-like shape of the distribution we are taking an average of):
    histosRing.add("Ring/LeadJet/hRingObservableCounts", "Counts vs <#it{R}>_{jet};<#it{R}>; Counts", kTH1D, {axisConfigurations.axisRingCounts});
    // For the leading particle:
    histosRing.add("Ring/LeadP/pRingObservableIntegrated", "Integrated <#it{R}> (LeadP); ;<#it{R}>", kTProfile, {{1, -0.5, 0.5}});
    histosRing.add("Ring/LeadP/hRingObservableCounts", "Counts vs <#it{R}>_{LeadP};<#it{R}>; Counts", kTH1D, {axisConfigurations.axisRingCounts});
    // For the second-to-leading jet:
    histosRing.add("Ring/SubJet/pRingObservableIntegrated", "Integrated <#it{R}> (SubJet); ;<#it{R}>", kTProfile, {{1, -0.5, 0.5}});
    histosRing.add("Ring/SubJet/hRingObservableCounts", "Counts vs <#it{R}>_{SubJet};<#it{R}>; Counts", kTH1D, {axisConfigurations.axisRingCounts});

    // ===============================
    // AngularDependence/
    // ===============================
    histosRing.add("Ring/LeadJet/AngularDependence/pRingObservableDeltaPhi", "<#it{R}> vs #Delta#varphi_{jet};#Delta#varphi_{jet};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaPhi});
    histosRing.add("Ring/LeadJet/AngularDependence/pRingObservableDeltaTheta", "<#it{R}> vs #Delta#theta_{jet};#Delta#theta_{jet};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaTheta});
    histosRing.add("Ring/LeadJet/AngularDependence/pRingObservableProxyPhi", "<#it{R}> vs #varphi_{jet};#varphi_{jet};<#it{R}>", kTProfile, {axisConfigurations.axisPhi});
    histosRing.add("Ring/LeadJet/AngularDependence/pRingObservablePhiLambda", "<#it{R}> vs #varphi_{#Lambda};#varphi_{#Lambda};<#it{R}>", kTProfile, {axisConfigurations.axisPhi});
    histosRing.add("Ring/LeadP/AngularDependence/pRingObservableDeltaPhi", "<#it{R}> vs #Delta#varphi_{LeadP};#Delta#varphi_{LeadP};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaPhi});
    histosRing.add("Ring/LeadP/AngularDependence/pRingObservableDeltaTheta", "<#it{R}> vs #Delta#theta_{LeadP};#Delta#theta_{LeadP};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaTheta});
    histosRing.add("Ring/SubJet/AngularDependence/pRingObservableDeltaPhi", "<#it{R}> vs #Delta#varphi_{SubJet};#Delta#varphi_{SubJet};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaPhi});
    histosRing.add("Ring/SubJet/AngularDependence/pRingObservableDeltaTheta", "<#it{R}> vs #Delta#theta_{SubJet};#Delta#theta_{SubJet};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaTheta});

    // ===============================
    // PtDependence/ (Lambda pT and proxy pT)
    // ===============================
    histosRing.add("Ring/LeadJet/PtDependence/pRingObservableLambdaPt", "<#it{R}> vs #it{p}_{T}^{#Lambda};#it{p}_{T}^{#Lambda} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisPt});
    histosRing.add("Ring/LeadP/PtDependence/pRingObservableLambdaPt", "<#it{R}> vs #it{p}_{T}^{#Lambda} (LeadP);#it{p}_{T}^{#Lambda} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisPt});
    histosRing.add("Ring/SubJet/PtDependence/pRingObservableLambdaPt", "<#it{R}> vs #it{p}_{T}^{#Lambda} (SubJet);#it{p}_{T}^{#Lambda} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisPt});
    // 2D TProfiles (Lambda correlations), with their counters:
    histosRing.add("Ring/LeadJet/PtDependence/p2dRingObservableDeltaPhiVsLambdaPt", "<#it{R}> vs #Delta#varphi_{jet} vs #it{p}_{T}^{#Lambda};#Delta#varphi_{jet};#it{p}_{T}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisPt});
    histosRing.add("Ring/LeadJet/PtDependence/p2dRingObservableDeltaThetaVsLambdaPt", "<#it{R}> vs #Delta#theta_{jet} vs #it{p}_{T}^{#Lambda};#Delta#theta_{jet};#it{p}_{T}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisPt});
    histosRing.add("Ring/LeadJet/PtDependence/h2dDeltaPhiVsLambdaPt", "#Delta#varphi_{jet} vs #Lambda #it{p}_{T};#Delta#varphi_{jet};#it{p}_{T}^{#Lambda} (GeV/c)", kTH2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisPt});
    histosRing.add("Ring/LeadJet/PtDependence/h2dDeltaThetaVsLambdaPt", "#Delta#theta_{jet} vs #Lambda #it{p}_{T};#Delta#theta_{jet};#it{p}_{T}^{#Lambda} (GeV/c)", kTH2D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisPt});
    // 2D TProfiles (Jet correlations):
    histosRing.add("Ring/LeadJet/PtDependence/p2dRingObservableDeltaPhiVsProxyPt", "<#it{R}> vs #Delta#varphi_{jet} vs Lead Jet #it{p}_{T};#Delta#varphi_{jet};#it{p}_{T}^{LeadJet} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisJetPt});
    histosRing.add("Ring/LeadJet/PtDependence/p2dRingObservableDeltaThetaVsProxyPt", "<#it{R}> vs #Delta#theta_{jet} vs Lead Jet #it{p}_{T};#Delta#theta_{jet};#it{p}_{T}^{LeadJet} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisJetPt});
    // Ring vs Jet proxy pT:
    histosRing.add("Ring/LeadJet/PtDependence/pRingVsProxyPt", "<#it{R}> vs Jet #it{p}_{T};#it{p}_{T}^{Jet} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
    histosRing.add("Ring/LeadP/PtDependence/pRingVsProxyPt", "<#it{R}> vs LeadP #it{p}_{T};#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
    histosRing.add("Ring/SubJet/PtDependence/pRingVsProxyPt", "<#it{R}> vs SubJet #it{p}_{T};#it{p}_{T}^{SubJet} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
    // Splitting into positive and negative eta contributions:
    histosRing.add("Ring/LeadJet/PtDependence/p2dRingVsProxyPtVsProxyEta", "<#it{R}> vs Jet #it{p}_{T} vs #eta_{Jet};#it{p}_{T}^{Jet} (GeV/c);#eta_{Jet};<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, {2, -0.9, 0.9}});
    histosRing.add("Ring/LeadP/PtDependence/p2dRingVsProxyPtVsProxyEta", "<#it{R}> vs LeadP #it{p}_{T} vs #eta_{LeadP};#it{p}_{T}^{LeadP} (GeV/c);#eta_{LeadP};<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, {2, -0.9, 0.9}});
    histosRing.add("Ring/SubJet/PtDependence/p2dRingVsProxyPtVsProxyEta", "<#it{R}> vs SubJet #it{p}_{T} vs #eta_{SubJet};#it{p}_{T}^{SubJet} (GeV/c);#eta_{SubJet};<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, {2, -0.9, 0.9}});
    // For each Lambda's eta:
    histosRing.add("Ring/LeadJet/PtDependence/p2dRingVsProxyPtVsEtaV0", "<#it{R}> vs Jet #it{p}_{T} vs #eta_{V0};#it{p}_{T}^{Jet} (GeV/c);#eta_{V0};<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, {2, -0.9, 0.9}});
    histosRing.add("Ring/LeadP/PtDependence/p2dRingVsProxyPtVsEtaV0", "<#it{R}> vs LeadP #it{p}_{T} vs #eta_{V0};#it{p}_{T}^{LeadP} (GeV/c);#eta_{V0};<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, {2, -0.9, 0.9}});
    histosRing.add("Ring/SubJet/PtDependence/p2dRingVsProxyPtVsEtaV0", "<#it{R}> vs SubJet #it{p}_{T} vs #eta_{V0};#it{p}_{T}^{SubJet} (GeV/c);#eta_{V0};<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, {2, -0.9, 0.9}});
    // Rasterizing, only for LeadP the TProfile2D into two TProfile 1Ds (easier to draw with "same")
    histosRing.add("Ring/LeadP/PtDependence/pRingVsProxyPt_PosProxyEta", "<#it{R}> vs LeadP #it{p}_{T} (#eta_{LeadP}>0);#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
    histosRing.add("Ring/LeadP/PtDependence/pRingVsProxyPt_NegProxyEta", "<#it{R}> vs LeadP #it{p}_{T} (#eta_{LeadP}<0);#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
    // V0 eta:
    histosRing.add("Ring/LeadP/PtDependence/pRingVsProxyPt_PosEtaV0", "<#it{R}> vs LeadP #it{p}_{T} (#eta_{V0}>0);#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
    histosRing.add("Ring/LeadP/PtDependence/pRingVsProxyPt_NegEtaV0", "<#it{R}> vs LeadP #it{p}_{T} (#eta_{V0}<0);#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
    // Looking at V0Eta and JetEta combinations (only for LeadP):
    histosRing.add("Ring/LeadP/PtDependence/pRingVsProxyPt_PosProxyEta_PosEtaV0", "<#it{R}> vs LeadP #it{p}_{T} (#eta_{LeadP}>0, #eta_{V0}>0);#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
    histosRing.add("Ring/LeadP/PtDependence/pRingVsProxyPt_NegProxyEta_PosEtaV0", "<#it{R}> vs LeadP #it{p}_{T} (#eta_{LeadP}<0, #eta_{V0}>0);#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
    histosRing.add("Ring/LeadP/PtDependence/pRingVsProxyPt_PosProxyEta_NegEtaV0", "<#it{R}> vs LeadP #it{p}_{T} (#eta_{LeadP}>0, #eta_{V0}<0);#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
    histosRing.add("Ring/LeadP/PtDependence/pRingVsProxyPt_NegProxyEta_NegEtaV0", "<#it{R}> vs LeadP #it{p}_{T} (#eta_{LeadP}<0, #eta_{V0}<0);#it{p}_{T}^{LeadP} (GeV/c);<#it{R}>", kTProfile, {axisConfigurations.axisJetPt});
    // Integrated <R> Vs ProxyPt Vs Centrality:
    histosRing.add("Ring/LeadJet/PtDependence/p2dRingVsProxyPtVsCentrality", "<#it{R}> vs Jet #it{p}_{T} vs Centrality;#it{p}_{T}^{Jet} (GeV/c);Centrality(%);<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, axisConfigurations.axisCentrality});
    histosRing.add("Ring/LeadP/PtDependence/p2dRingVsProxyPtVsCentrality", "<#it{R}> vs LeadP #it{p}_{T} vs Centrality;#it{p}_{T}^{LeadP} (GeV/c);Centrality(%);<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, axisConfigurations.axisCentrality});
    histosRing.add("Ring/SubJet/PtDependence/p2dRingVsProxyPtVsCentrality", "<#it{R}> vs SubJet #it{p}_{T} vs Centrality;#it{p}_{T}^{SubJet} (GeV/c);Centrality(%);<#it{R}>", kTProfile2D, {axisConfigurations.axisJetPt, axisConfigurations.axisCentrality});

    // ===============================
    // EtaDependence/
    // ===============================
    // Understanding eta dependence seen in pRingEtaCuts:
    histosRing.add("Ring/LeadJet/EtaDependence/pRingObservableEtaLambda", "<#it{R}> vs #eta_{#Lambda};#eta_{#Lambda};<#it{R}>", kTProfile, {axisConfigurations.axisV0EtaCoarse});
    histosRing.add("Ring/LeadJet/EtaDependence/pRingObservableProxyEta", "<#it{R}> vs #eta_{Jet};#eta_{Jet};<#it{R}>", kTProfile, {axisConfigurations.axisEtaCoarse});
    histosRing.add("Ring/LeadJet/EtaDependence/pRingObservableProxyEtaHighEtaRes", "<#it{R}> vs #eta_{Jet};#eta_{Jet};<#it{R}>", kTProfile, {axisConfigurations.axisEta});
    histosRing.add("Ring/SubJet/EtaDependence/pRingObservableEtaLambda", "<#it{R}> vs #eta_{#Lambda} (SubJet);#eta_{#Lambda};<#it{R}>", kTProfile, {axisConfigurations.axisV0EtaCoarse});
    histosRing.add("Ring/SubJet/EtaDependence/pRingObservableProxyEta", "<#it{R}> vs #eta_{SubJet};#eta_{SubJet};<#it{R}>", kTProfile, {axisConfigurations.axisEtaCoarse});
    histosRing.add("Ring/LeadP/EtaDependence/pRingObservableEtaLambda", "<#it{R}> vs #eta_{#Lambda} (LeadP);#eta_{#Lambda};<#it{R}>", kTProfile, {axisConfigurations.axisV0EtaCoarse});
    histosRing.add("Ring/LeadP/EtaDependence/pRingObservableProxyEta", "<#it{R}> vs #eta_{LeadP};#eta_{LeadP};<#it{R}>", kTProfile, {axisConfigurations.axisEtaCoarse});
    histosRing.add("Ring/LeadP/EtaDependence/pRingObservableProxyEtaHighEtaRes", "<#it{R}> vs #eta_{LeadP};#eta_{LeadP};<#it{R}>", kTProfile, {axisConfigurations.axisEta});
    // TProfile2D: <R> vs Eta Lambda vs Eta Jet (Understanding eta dependence seen in pRingEtaCuts)
    histosRing.add("Ring/LeadJet/EtaDependence/hCounterEtaLambdaMinusProxyEta", "N_{V0s} vs #eta_{#Lambda} - #eta_{Jet};#eta_{#Lambda} - #eta_{Jet}; N_{V0s}", kTH1D, {axisConfigurations.axisDeltaEtaCoarse});
    histosRing.add("Ring/LeadJet/EtaDependence/pRingObservableEtaLambdaMinusProxyEta", "<#it{R}> vs #eta_{#Lambda} - #eta_{Jet};#eta_{#Lambda} - #eta_{Jet};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaEtaCoarse});
    histosRing.add("Ring/LeadJet/EtaDependence/p2dRingObservableEtaLambdaVsProxyEta", "<#it{R}> vs #eta_{#Lambda} vs #eta_{Jet};#eta_{#Lambda};#eta_{Jet};<#it{R}>", kTProfile2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisEtaCoarse});
    histosRing.add("Ring/LeadJet/EtaDependence/p2dRingObservableEtaLambdaVsProxyEta_FineBins", "<#it{R}> vs #eta_{#Lambda} vs #eta_{Jet} (fine bins);#eta_{#Lambda};#eta_{Jet};<#it{R}>", kTProfile2D, {axisConfigurations.axisV0Eta, axisConfigurations.axisEta});
    histosRing.add("Ring/LeadP/EtaDependence/p2dRingObservableEtaLambdaVsProxyEta", "<#it{R}> vs #eta_{#Lambda} vs #eta_{LeadP};#eta_{#Lambda};#eta_{LeadP};<#it{R}>", kTProfile2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisEtaCoarse});
    histosRing.add("Ring/SubJet/EtaDependence/p2dRingObservableEtaLambdaVsProxyEta", "<#it{R}> vs #eta_{#Lambda} vs #eta_{SubJet};#eta_{#Lambda};#eta_{SubJet};<#it{R}>", kTProfile2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisEtaCoarse});
    // Counters for these histograms, instead of only TProfile2Ds:
    histosRing.add("Ring/LeadJet/EtaDependence/h2dCounterEtaLambdaVsProxyEta", "Counts, #eta_{#Lambda} vs #eta_{Jet};#eta_{#Lambda};#eta_{Jet};Counts", kTH2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisEtaCoarse});
    histosRing.add("Ring/LeadJet/EtaDependence/h2dCounterEtaLambdaVsProxyEta_FineBins", "Counts (fine bins), #eta_{#Lambda} vs #eta_{Jet};#eta_{#Lambda};#eta_{Jet};Counts", kTH2D, {axisConfigurations.axisV0Eta, axisConfigurations.axisEta});
    histosRing.add("Ring/LeadP/EtaDependence/h2dCounterEtaLambdaVsProxyEta", "Counts, #eta_{#Lambda} vs #eta_{LeadP};#eta_{#Lambda};#eta_{LeadP};Counts", kTH2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisEtaCoarse});
    histosRing.add("Ring/SubJet/EtaDependence/h2dCounterEtaLambdaVsProxyEta", "Counts, #eta_{#Lambda} vs #eta_{SubJet};#eta_{#Lambda};#eta_{SubJet};Counts", kTH2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisEtaCoarse});

    // ===============================
    // MassDependence/: quick glancing and signal extraction
    // ===============================
    // Simple invariant mass plot, in the signal extraction binning:
    histosRing.add("Ring/LeadJet/MassDependence/hMassSigExtract", "#Lambda Mass (Sig Extract);m_{p#pi} (GeV/c^{2});Counts", kTH1D, {axisConfigurations.axisLambdaMassSigExtract});
    // TProfile of ring vs mass (integrated in all phi, and properly normalized by N_Lambda):
    histosRing.add("Ring/LeadJet/MassDependence/pRingObservableMass", "<#it{R}> vs Mass;m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
    histosRing.add("Ring/LeadP/MassDependence/pRingObservableMass", "<#it{R}> vs Mass (LeadP);m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
    histosRing.add("Ring/SubJet/MassDependence/pRingObservableMass", "<#it{R}> vs Mass (SubJet);m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
    // TProfile2D: <R> vs Mass (DeltaPhi)
    histosRing.add("Ring/LeadJet/MassDependence/p2dRingObservableDeltaPhiVsMass", "<#it{R}> vs #Delta#varphi_{jet} vs Mass;#Delta#varphi_{jet};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisLambdaMassSigExtract});
    // TProfile2D: <R> vs Mass (DeltaTheta)
    histosRing.add("Ring/LeadJet/MassDependence/p2dRingObservableDeltaThetaVsMass", "<#it{R}> vs #Delta#theta_{jet} vs Mass;#Delta#theta_{jet};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisLambdaMassSigExtract});
    // --- 2D counters: Angle vs Mass ---
    histosRing.add("Ring/LeadJet/MassDependence/h2dDeltaPhiVsMass", "#Delta#varphi_{jet} vs Mass;#Delta#varphi_{jet};m_{p#pi} (GeV/c^{2})", kTH2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisLambdaMassSigExtract});
    histosRing.add("Ring/LeadJet/MassDependence/h2dDeltaThetaVsMass", "#Delta#theta_{jet} vs Mass;#Delta#theta_{jet};m_{p#pi} (GeV/c^{2})", kTH2D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisLambdaMassSigExtract});
    // (No 3D counters here: the bin entries of the p3d profiles in SigExtract3DInput/ already count the same V0s)
    // TProfile2D: <R> vs Mass (EtaLambda):
    histosRing.add("Ring/LeadJet/MassDependence/p2dRingObservableEtaLambdaVsMass", "<#it{R}> vs #eta_{#Lambda} vs Mass;#eta_{#Lambda};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisLambdaMassSigExtract});
    // TProfile2D: <R> vs EtaProxy Vs Mass
    histosRing.add("Ring/LeadJet/MassDependence/p2dRingObservableProxyEtaVsMass", "<#it{R}> vs #eta_{jet} vs Mass;#eta_{jet};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
    histosRing.add("Ring/LeadP/MassDependence/p2dRingObservableProxyEtaVsMass", "<#it{R}> vs #eta_{LeadP} vs Mass;#eta_{LeadP};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
    histosRing.add("Ring/SubJet/MassDependence/p2dRingObservableProxyEtaVsMass", "<#it{R}> vs #eta_{SubJet} vs Mass;#eta_{SubJet};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
    // Auxiliary counters for EtaProxy Vs Mass:
    histosRing.add("Ring/LeadJet/MassDependence/h2dCounterProxyEtaVsMass", "V0 Counter vs #eta_{jet} vs Mass;#eta_{jet};m_{p#pi} (GeV/c^{2});Counter", kTH2D, {axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
    histosRing.add("Ring/LeadP/MassDependence/h2dCounterProxyEtaVsMass", "V0 Counter vs #eta_{LeadP} vs Mass;#eta_{LeadP};m_{p#pi} (GeV/c^{2});Counter", kTH2D, {axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
    histosRing.add("Ring/SubJet/MassDependence/h2dCounterProxyEtaVsMass", "V0 Counter vs #eta_{SubJet} vs Mass;#eta_{SubJet};m_{p#pi} (GeV/c^{2});Counter", kTH2D, {axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
    // --- TProfile2D: <R> vs Mass vs Centrality ---
    histosRing.add("Ring/LeadJet/MassDependence/p2dRingObservableMassVsCent", "<#it{R}> vs Mass vs Centrality;m_{p#pi} (GeV/c^{2});Centrality (%);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisCentrality});

    // ===============================
    // EventLevelDependence/
    // ===============================
    // --- TProfile1D: Integrated <R> vs Centrality:
    histosRing.add("Ring/LeadJet/EventLevelDependence/pRingVsCentrality", "<#it{R}> vs Centrality;Centrality (%);<#it{R}>", kTProfile, {axisConfigurations.axisCentrality});
    // For the Zvtx dependence:
    histosRing.add("Ring/LeadJet/EventLevelDependence/pRingObservablePVz", "<#it{R}>_{LeadJet} vs PVz;PVz (cm);<#it{R}>", kTProfile, {axisConfigurations.axisPVzCoarse});
    histosRing.add("Ring/SubJet/EventLevelDependence/pRingObservablePVz", "<#it{R}>_{SubLead} vs PVz;PVz (cm);<#it{R}>", kTProfile, {axisConfigurations.axisPVzCoarse});
    histosRing.add("Ring/LeadP/EventLevelDependence/pRingObservablePVz", "<#it{R}>_{LeadP} vs PVz;PVz (cm);<#it{R}>", kTProfile, {axisConfigurations.axisPVzCoarse});

    // ===============================
    // RingMaps/: ring observable, single scalar 2D profile:
    // ===============================
    histosRing.add("Ring/LeadJet/RingMaps/p2dRingObservableVsPxPy", "<#it{R}> vs (p_{x}^{#Lambda},p_{y}^{#Lambda});p_{x}^{#Lambda} (GeV/c);p_{y}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
    histosRing.add("Ring/LeadJet/RingMaps/p2dRingObservableVsPzPx", "<#it{R}> vs (p_{z}^{#Lambda},p_{x}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{x}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPRot});
    histosRing.add("Ring/LeadJet/RingMaps/p2dRingObservableVsPyPz", "<#it{R}> vs (p_{y}^{#Lambda},p_{z}^{#Lambda});p_{y}^{#Lambda} (GeV/c);p_{z}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPz});
    // On the AEE planes (no need to rotate before calculating <R>, as it is invariant by rotation -- the only change is in the XY plane it is plotted in):
    histosRing.add("Ring/LeadJet/RingMaps/p2dRingObservableVsPxAeePyAee", "<#it{R}> vs (p_{x,AEE}^{#Lambda},p_{y,AEE}^{#Lambda});p_{x,AEE}^{#Lambda} (GeV/c);p_{y,AEE}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
    histosRing.add("Ring/LeadJet/RingMaps/p2dRingObservableVsPzPxAee", "<#it{R}> vs (p_{z}^{#Lambda},p_{x,AEE}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{x,AEE}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPRot});
    histosRing.add("Ring/LeadJet/RingMaps/p2dRingObservableVsPyAeePz", "<#it{R}> vs (p_{y,AEE}^{#Lambda},p_{z}^{#Lambda});p_{y,AEE}^{#Lambda} (GeV/c);p_{z}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPz});
    // On the jet-frame planes:
    histosRing.add("Ring/LeadJet/RingMaps/p2dRingObservableVsPxPyPrimeJet", "<#it{R}> vs (p_{x'Jet}^{#Lambda},p_{y'Jet}^{#Lambda});p_{x'Jet}^{#Lambda} (GeV/c);p_{y'Jet}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
    histosRing.add("Ring/LeadJet/RingMaps/p2dRingObservableVsPzPxPrimeJet", "<#it{R}> vs (p_{z'Jet}^{#Lambda},p_{x'Jet}^{#Lambda});p_{z'Jet}^{#Lambda} (GeV/c);p_{x'Jet}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
    histosRing.add("Ring/LeadJet/RingMaps/p2dRingObservableVsPyPzPrimeJet", "<#it{R}> vs (p_{y'Jet}^{#Lambda},p_{z'Jet}^{#Lambda});p_{y'Jet}^{#Lambda} (GeV/c);p_{z'Jet}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
    // For LeadP estimators:
    histosRing.add("Ring/LeadP/RingMaps/p2dRingObservableVsPxPy", "<#it{R}>_{LeadP} vs (p_{x}^{#Lambda},p_{y}^{#Lambda});p_{x}^{#Lambda} (GeV/c);p_{y}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
    histosRing.add("Ring/LeadP/RingMaps/p2dRingObservableVsPzPx", "<#it{R}>_{LeadP} vs (p_{z}^{#Lambda},p_{x}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{x}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPRot});
    histosRing.add("Ring/LeadP/RingMaps/p2dRingObservableVsPyPz", "<#it{R}>_{LeadP} vs (p_{y}^{#Lambda},p_{z}^{#Lambda});p_{y}^{#Lambda} (GeV/c);p_{z}^{#Lambda} (GeV/c);<#it{R}>", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPz});

    // ===============================
    // RingKernel/: ring projection kernel
    // The two variables <R> depends on analytically, plus the Lambda direction cosine in the 3D version,
    // which is what lets the eta_Lambda sample be symmetrised downstream instead of at fill time.
    // ===============================
    // Binned in the two variables the projection is analytically linear in -- cos(DeltaTheta) and the
    // proxy direction cosine t_z -- rather than in DeltaTheta and eta. Any fitting of the resulting
    // surface belongs downstream, in the post-processing macros.
    histosRing.add("Ring/LeadJet/RingKernel/p2dRingObservableCosDeltaThetaVsProxyZ", "<#it{R}> vs (cos#Delta#theta_{jet},#hat{t}_{z});cos#Delta#theta_{jet};#hat{t}_{z};<#it{R}>", kTProfile2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisJetZ});
    histosRing.add("Ring/LeadJet/RingKernel/h2dCountsCosDeltaThetaVsProxyZ", "Counts vs (cos#Delta#theta_{jet},#hat{t}_{z});cos#Delta#theta_{jet};#hat{t}_{z};Counts", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisJetZ});
    histosRing.add("Ring/LeadJet/RingKernel/p3dRingObservableCosDeltaThetaVsProxyZVsLambdaZ", "<#it{R}> vs (cos#Delta#theta_{jet},#hat{t}_{z},cos#theta_{#Lambda});cos#Delta#theta_{jet};#hat{t}_{z};cos#theta_{#Lambda}", kTProfile3D, {axisConfigurations.axisCosThetaCoarse, axisConfigurations.axisJetZ, axisConfigurations.axisLambdaZ});

    // LeadP proxy. B_phi knows nothing about the reference axis, so this must return the same kernel as the jet version -- with a wider t_z lever.
    histosRing.add("Ring/LeadP/RingKernel/p2dRingObservableCosDeltaThetaVsProxyZ", "<#it{R}>_{LeadP} vs (cos#Delta#theta_{LeadP},#hat{t}_{z}^{LeadP});cos#Delta#theta_{LeadP};#hat{t}_{z}^{LeadP};<#it{R}>", kTProfile2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisLeadPZ});
    histosRing.add("Ring/LeadP/RingKernel/h2dCountsCosDeltaThetaVsProxyZ", "Counts vs (cos#Delta#theta_{LeadP},#hat{t}_{z}^{LeadP});cos#Delta#theta_{LeadP};#hat{t}_{z}^{LeadP};Counts", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisLeadPZ});
    histosRing.add("Ring/LeadP/RingKernel/p3dRingObservableCosDeltaThetaVsProxyZVsLambdaZ", "<#it{R}>_{LeadP} vs (cos#Delta#theta_{LeadP},#hat{t}_{z}^{LeadP},cos#theta_{#Lambda});cos#Delta#theta_{LeadP};#hat{t}_{z}^{LeadP};cos#theta_{#Lambda}", kTProfile3D, {axisConfigurations.axisCosThetaCoarse, axisConfigurations.axisLeadPZ, axisConfigurations.axisLambdaZ});

    // KappaEff moments, per proxy and vs mass for signal extraction:
    // With u the unitless ring (R without its prefactor), v the unitless full ring and w = e \cdot n the projection weight,
    // kappa = 3 <u v>/<w> -- the cross moment, exact for any projection. Full ring: u = v and w = 1. R_z and R_z^T: w = n_z^2.
    for (const std::string& proxy : {std::string("LeadJet"), std::string("LeadP"), std::string("SubJet")}) {
      histosRing.add(("Ring/" + proxy + "/KappaEff/pKappaNumVsMass").c_str(), ("<u v> vs Mass (" + proxy + ");m_{p#pi} (GeV/c^{2});<u v>").c_str(), kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
      histosRing.add(("Ring/" + proxy + "/KappaEff/pKappaDenVsMass").c_str(), ("<w> vs Mass (" + proxy + ");m_{p#pi} (GeV/c^{2});<w>").c_str(), kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
      histosRing.add(("Ring/" + proxy + "/KappaEff/pKappaNumTimesDenVsMass").c_str(), ("<u v w> vs Mass (" + proxy + ");m_{p#pi} (GeV/c^{2});<u v w>").c_str(), kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
      histosRing.add(("Ring/" + proxy + "/KappaEff/pRingTimesDenVsMass").c_str(), ("<#it{R} w> vs Mass (" + proxy + ");m_{p#pi} (GeV/c^{2});<#it{R} w>").c_str(), kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
      // The same moments per CheapSigExtract region, from v0InMassWindow and v0InMassPeak: bin 1 sideband, bin 2 peak, underflow otherwise
      histosRing.add(("Ring/" + proxy + "/KappaEff/pRingVsMassRegion").c_str(), ("<#it{R}> per mass region (" + proxy + ");mass region (1 sideband, 2 peak);<#it{R}>").c_str(), kTProfile, {{2, 0., 2.}});
      histosRing.add(("Ring/" + proxy + "/KappaEff/pKappaNumVsMassRegion").c_str(), ("<u v> per mass region (" + proxy + ");mass region (1 sideband, 2 peak);<u v>").c_str(), kTProfile, {{2, 0., 2.}});
      histosRing.add(("Ring/" + proxy + "/KappaEff/pKappaDenVsMassRegion").c_str(), ("<w> per mass region (" + proxy + ");mass region (1 sideband, 2 peak);<w>").c_str(), kTProfile, {{2, 0., 2.}});
      histosRing.add(("Ring/" + proxy + "/KappaEff/pKappaNumTimesDenVsMassRegion").c_str(), ("<u v w> per mass region (" + proxy + ");mass region (1 sideband, 2 peak);<u v w>").c_str(), kTProfile, {{2, 0., 2.}});
      histosRing.add(("Ring/" + proxy + "/KappaEff/pRingTimesDenVsMassRegion").c_str(), ("<#it{R} w> per mass region (" + proxy + ");mass region (1 sideband, 2 peak);<#it{R} w>").c_str(), kTProfile, {{2, 0., 2.}});

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
          histosRing.add(("Ring/" + proxy + "/KappaEff/Differential/p2d" + mom.name + "Vs" + var.name + "VsMassRegion").c_str(),
                         (mom.title + " vs " + var.name + " per mass region (" + proxy + ");" + var.title + ";mass region (1 sideband, 2 peak);" + mom.title).c_str(),
                         kTProfile2D, {var.axis, {2, 0., 2.}});
        }
      }
    }

    // ===============================
    // V0ProxyPairQA/: angle and pT distributions of the V0s paired with a proxy
    // (No mass dependency -- useful to check kinematic sculpting from cuts)
    // (Not the KinematicsQA/ population: here we look into the V0s of events that also have a proxy -- thus we are "pairing" V0s and proxies)
    // ===============================
    histosRing.add("Ring/LeadJet/V0ProxyPairQA/hDeltaPhi", "#Delta#varphi_{jet};#Delta#varphi_{jet};Counts", kTH1D, {axisConfigurations.axisDeltaPhi});
    histosRing.add("Ring/LeadJet/V0ProxyPairQA/h2dDeltaPhiVsDeltaEta", "#Delta#varphi_{jet};#Delta#varphi_{jet}; #eta_{#Lambda}-#eta_{Jet};Counts", kTH2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDeltaEtaCoarse});
    histosRing.add("Ring/LeadJet/V0ProxyPairQA/h2dDeltaPhiVsProxyPhi", "#Delta#varphi_{jet} vs #varphi_{Jet};#Delta#varphi_{jet};#varphi_{Jet};Counts", kTH2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisPhi});
    histosRing.add("Ring/LeadJet/V0ProxyPairQA/hDeltaTheta", "#Delta#theta_{jet};#Delta#theta_{jet};Counts", kTH1D, {axisConfigurations.axisDeltaTheta});
    histosRing.add("Ring/LeadJet/V0ProxyPairQA/hCosDeltaTheta", "cos(#Delta#theta_{jet});cos(#Delta#theta_{jet});Counts", kTH1D, {axisConfigurations.axisCosTheta}); // Should actually be flat due to the geometry

    histosRing.add("Ring/LeadP/V0ProxyPairQA/hDeltaPhi", "#Delta#varphi_{LeadP};#Delta#varphi_{LeadP};Counts", kTH1D, {axisConfigurations.axisDeltaPhi});
    histosRing.add("Ring/LeadP/V0ProxyPairQA/hDeltaTheta", "#Delta#theta_{LeadP};#Delta#theta_{LeadP};Counts", kTH1D, {axisConfigurations.axisDeltaTheta});
    histosRing.add("Ring/LeadP/V0ProxyPairQA/hCosDeltaTheta", "cos(#Delta#theta_{LeadP});cos(#Delta#theta_{LeadP});Counts", kTH1D, {axisConfigurations.axisCosTheta}); // Should actually be flat due to the geometry
    histosRing.add("Ring/SubJet/V0ProxyPairQA/hDeltaPhi", "#Delta#varphi_{SubJet};#Delta#varphi_{SubJet};Counts", kTH1D, {axisConfigurations.axisDeltaPhi});
    histosRing.add("Ring/SubJet/V0ProxyPairQA/hDeltaTheta", "#Delta#theta_{SubJet};#Delta#theta_{SubJet};Counts", kTH1D, {axisConfigurations.axisDeltaTheta});
    histosRing.add("Ring/SubJet/V0ProxyPairQA/hCosDeltaTheta", "cos(#Delta#theta_{SubJet});cos(#Delta#theta_{SubJet});Counts", kTH1D, {axisConfigurations.axisCosTheta}); // Should actually be flat due to the geometry
    // And some counters to be aware of the amount of Lambdas (and jets) in each pT interval:
    histosRing.add("Ring/LeadJet/V0ProxyPairQA/hPairedProxyPt", "Jet #it{p}_{T};#it{p}_{T}^{Jet} (GeV/c);Counts", kTH1D, {axisConfigurations.axisJetPt});
    histosRing.add("Ring/LeadP/V0ProxyPairQA/hPairedProxyPt", "LeadP #it{p}_{T};#it{p}_{T}^{LeadP} (GeV/c);Counts", kTH1D, {axisConfigurations.axisJetPt});
    histosRing.add("Ring/SubJet/V0ProxyPairQA/hPairedProxyPt", "SubJet #it{p}_{T};#it{p}_{T}^{SubJet} (GeV/c);Counts", kTH1D, {axisConfigurations.axisJetPt});
    histosRing.add("Ring/LeadJet/V0ProxyPairQA/hPairedV0Pt", "#Lambda #it{p}_{T};#it{p}_{T}^{#Lambda} (GeV/c);Counts", kTH1D, {axisConfigurations.axisPt});
    // Simple invariant mass plot for QA:
    histosRing.add("Ring/LeadJet/V0ProxyPairQA/hPairedV0Mass", "#Lambda Mass;m_{p#pi} (GeV/c^{2});Counts", kTH1D, {axisConfigurations.axisLambdaMass});

    // ===============================
    // SigExtract3DInput/: multi-dimensional histograms for signal extraction, not meant to be viewed by hand
    // (Mass-dependent polarization extraction)
    // ===============================
    // --- TProfile3D: <R> vs DeltaPhi vs Mass vs LambdaPt ---
    histosRing.add("Ring/LeadJet/SigExtract3DInput/p3dRingObservableDeltaPhiVsMassVsLambdaPt", "<#it{R}> vs #Delta#varphi_{jet} vs Mass vs #it{p}_{T}^{#Lambda};#Delta#varphi_{jet};m_{p#pi} (GeV/c^{2});#it{p}_{T}^{#Lambda} (GeV/c)", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisPtSigExtract});
    // --- TProfile3D: <R> vs DeltaTheta vs Mass vs LambdaPt ---
    histosRing.add("Ring/LeadJet/SigExtract3DInput/p3dRingObservableDeltaThetaVsMassVsLambdaPt", "<#it{R}> vs #Delta#theta_{jet} vs Mass vs #it{p}_{T}^{#Lambda};#Delta#theta_{jet};m_{p#pi} (GeV/c^{2});#it{p}_{T}^{#Lambda} (GeV/c)", kTProfile3D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisPtSigExtract});
    // --- TProfile3D: <R> vs DeltaPhi vs Mass vs LeadJetPt ---
    histosRing.add("Ring/LeadJet/SigExtract3DInput/p3dRingObservableDeltaPhiVsMassVsProxyPt", "<#it{R}> vs #Delta#varphi_{jet} vs Mass vs Lead Jet #it{p}_{T};#Delta#varphi_{jet};m_{p#pi} (GeV/c^{2});#it{p}_{T}^{LeadJet} (GeV/c)", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisJetPtSigExtract});
    // --- TProfile3D: <R> vs DeltaTheta vs Mass vs LeadJetPt ---
    histosRing.add("Ring/LeadJet/SigExtract3DInput/p3dRingObservableDeltaThetaVsMassVsProxyPt", "<#it{R}> vs #Delta#theta_{jet} vs Mass vs Lead Jet #it{p}_{T};#Delta#theta_{jet};m_{p#pi} (GeV/c^{2});#it{p}_{T}^{LeadJet} (GeV/c)", kTProfile3D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisJetPtSigExtract});
    // --- TProfile3D: <R> vs DeltaPhi vs Mass vs Centrality ---
    histosRing.add("Ring/LeadJet/SigExtract3DInput/p3dRingObservableDeltaPhiVsMassVsCent", "<#it{R}> vs #Delta#varphi_{jet} vs Mass vs Centrality;#Delta#varphi_{jet};m_{p#pi} (GeV/c^{2});Centrality (%)", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisCentrality});
    // --- TProfile3D: <R> vs DeltaTheta vs Mass vs Centrality ---
    histosRing.add("Ring/LeadJet/SigExtract3DInput/p3dRingObservableDeltaThetaVsMassVsCent", "<#it{R}> vs #Delta#theta_{jet} vs Mass vs Centrality;#Delta#theta_{jet};m_{p#pi} (GeV/c^{2});Centrality (%)", kTProfile3D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisCentrality});

    // ===============================
    // RawPolarization/: polarization observable QAs
    // (not Ring: actual polarization!)
    // ===============================
    // Will implement these as TProfiles, as polarization is also a measure like P_Lambda = (3/\alpha_Lambda) * <p_{proton}>, so the error is similarly the SEM
    // ===============================
    // 1D TProfiles
    // ===============================
    histosRing.add("RawPolarization/pPxStarPhi", "<P_{#Lambda}^{*}>_{x} vs #varphi_{#Lambda};#varphi_{#Lambda};<P_{#Lambda}^{*}>_{x}", kTProfile, {axisConfigurations.axisDeltaPhi});
    histosRing.add("RawPolarization/pPyStarPhi", "<P_{#Lambda}^{*}>_{y} vs #varphi_{#Lambda};#varphi_{#Lambda};<P_{#Lambda}^{*}>_{y}", kTProfile, {axisConfigurations.axisDeltaPhi});
    histosRing.add("RawPolarization/pPzStarPhi", "<P_{#Lambda}^{*}>_{z} vs #varphi_{#Lambda};#varphi_{#Lambda};<P_{#Lambda}^{*}>_{z}", kTProfile, {axisConfigurations.axisDeltaPhi});
    histosRing.add("RawPolarization/LeadJet/pPxStarDeltaPhi", "<P_{#Lambda}^{*}>_{x} vs #Delta#varphi_{jet};#Delta#varphi_{jet};<P_{#Lambda}^{*}>_{x}", kTProfile, {axisConfigurations.axisDeltaPhi});
    histosRing.add("RawPolarization/LeadJet/pPyStarDeltaPhi", "<P_{#Lambda}^{*}>_{y} vs #Delta#varphi_{jet};#Delta#varphi_{jet};<P_{#Lambda}^{*}>_{y}", kTProfile, {axisConfigurations.axisDeltaPhi});
    histosRing.add("RawPolarization/LeadJet/pPzStarDeltaPhi", "<P_{#Lambda}^{*}>_{z} vs #Delta#varphi_{jet};#Delta#varphi_{jet};<P_{#Lambda}^{*}>_{z}", kTProfile, {axisConfigurations.axisDeltaPhi});
    // Profiles of polarization Vs AEE angle:
    // (there should be NO dependence on the AEE angle for the Z-axis polarization)
    histosRing.add("RawPolarization/pPxStarPhiAEE", "<P_{#Lambda}^{*}>_{x} vs AEE angle;#phi_{#Lambda}-#phi_{p}^{*};<P_{#Lambda}^{*}>_{x}", kTProfile, {axisConfigurations.axisDeltaPhi});
    histosRing.add("RawPolarization/pPyStarPhiAEE", "<P_{#Lambda}^{*}>_{y} vs AEE angle;#phi_{#Lambda}-#phi_{p}^{*};<P_{#Lambda}^{*}>_{y}", kTProfile, {axisConfigurations.axisDeltaPhi});
    histosRing.add("RawPolarization/pPzStarPhiAEE", "<P_{#Lambda}^{*}>_{z} vs AEE angle;#phi_{#Lambda}-#phi_{p}^{*};<P_{#Lambda}^{*}>_{z}", kTProfile, {axisConfigurations.axisDeltaPhi});
    // ===============================
    // 2D TProfiles (Lambda correlations)
    // ===============================
    histosRing.add("RawPolarization/LeadJet/p2dPxStarDeltaPhiVsLambdaPt", "<P_{#Lambda}^{*}>_{x} vs #Delta#varphi_{jet} vs #it{p}_{T}^{#Lambda};#Delta#varphi_{jet};#it{p}_{T}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{x}", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisPtSigExtract});
    histosRing.add("RawPolarization/LeadJet/p2dPyStarDeltaPhiVsLambdaPt", "<P_{#Lambda}^{*}>_{y} vs #Delta#varphi_{jet} vs #it{p}_{T}^{#Lambda};#Delta#varphi_{jet};#it{p}_{T}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{y}", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisPtSigExtract});
    histosRing.add("RawPolarization/LeadJet/p2dPzStarDeltaPhiVsLambdaPt", "<P_{#Lambda}^{*}>_{z} vs #Delta#varphi_{jet} vs #it{p}_{T}^{#Lambda};#Delta#varphi_{jet};#it{p}_{T}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{z}", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisPtSigExtract});

    // =========================================================================================
    // 2D vector-field profiles (+QA TH2D counters) of the polarization, in four coordinate systems:
    // (a companion code plots these as vector fields)
    // Frames are:
    // - Lab - the detector frame's PxPyPz
    // - Aee - XY rotated about \hat z so that \hat x_Aee = \hat p_{proton, Transverse}^*
    // - PrimeV0 - XY rotated about \hat z so that \hat x_V0 = \hat p_{Lambda, Transverse}
    // - PrimeJet - \hat z_PrimeJet = \hat JetDirection and \hat x_PrimeJet is built based on the lab's \hat z direction
    // (outside Ring/: with the kinematic-cut families gone there are no competing jet-direction definitions anymore, e.g. for the PrimeJet plane)
    // =========================================================================================
    // Lab frame:
    histosRing.add("RawPolarization/PolMaps/Lab/p2dPxStar_vsPxPy", "<P_{#Lambda}^{*}>_{x} vs (p_{x}^{#Lambda},p_{y}^{#Lambda});p_{x}^{#Lambda} (GeV/c);p_{y}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{x}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
    histosRing.add("RawPolarization/PolMaps/Lab/p2dPyStar_vsPxPy", "<P_{#Lambda}^{*}>_{y} vs (p_{x}^{#Lambda},p_{y}^{#Lambda});p_{x}^{#Lambda} (GeV/c);p_{y}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{y}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
    histosRing.add("RawPolarization/PolMaps/Lab/p2dPzStar_vsPxPy", "<P_{#Lambda}^{*}>_{z} vs (p_{x}^{#Lambda},p_{y}^{#Lambda});p_{x}^{#Lambda} (GeV/c);p_{y}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{z}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
    histosRing.add("RawPolarization/PolMaps/Lab/h2dCountsVsPxPy", "V0 Counts vs (p_{x}^{#Lambda},p_{y}^{#Lambda});p_{x}^{#Lambda} (GeV/c);p_{y}^{#Lambda} (GeV/c);Counts", kTH2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});

    histosRing.add("RawPolarization/PolMaps/Lab/p2dPxStar_vsPzPx", "<P_{#Lambda}^{*}>_{x} vs (p_{z}^{#Lambda},p_{x}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{x}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{x}", kTProfile2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPRot});
    histosRing.add("RawPolarization/PolMaps/Lab/p2dPyStar_vsPzPx", "<P_{#Lambda}^{*}>_{y} vs (p_{z}^{#Lambda},p_{x}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{x}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{y}", kTProfile2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPRot});
    histosRing.add("RawPolarization/PolMaps/Lab/p2dPzStar_vsPzPx", "<P_{#Lambda}^{*}>_{z} vs (p_{z}^{#Lambda},p_{x}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{x}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{z}", kTProfile2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPRot});
    histosRing.add("RawPolarization/PolMaps/Lab/h2dCountsVsPzPx", "V0 Counts vs (p_{z}^{#Lambda},p_{x}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{x}^{#Lambda} (GeV/c);Counts", kTH2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPRot});

    histosRing.add("RawPolarization/PolMaps/Lab/p2dPxStar_vsPyPz", "<P_{#Lambda}^{*}>_{x} vs (p_{y}^{#Lambda},p_{z}^{#Lambda});p_{y}^{#Lambda} (GeV/c);p_{z}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{x}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPz});
    histosRing.add("RawPolarization/PolMaps/Lab/p2dPyStar_vsPyPz", "<P_{#Lambda}^{*}>_{y} vs (p_{y}^{#Lambda},p_{z}^{#Lambda});p_{y}^{#Lambda} (GeV/c);p_{z}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{y}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPz});
    histosRing.add("RawPolarization/PolMaps/Lab/p2dPzStar_vsPyPz", "<P_{#Lambda}^{*}>_{z} vs (p_{y}^{#Lambda},p_{z}^{#Lambda});p_{y}^{#Lambda} (GeV/c);p_{z}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{z}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPz});
    histosRing.add("RawPolarization/PolMaps/Lab/h2dCountsVsPyPz", "V0 Counts vs (p_{y}^{#Lambda},p_{z}^{#Lambda});p_{y}^{#Lambda} (GeV/c);p_{z}^{#Lambda} (GeV/c);Counts", kTH2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPz});

    // Aee frame (coordinates where PhiAEE = 0 in direction \hat x'):
    // In this frame the transverse polarization is (|P*_T|, 0) by construction, so <P*_x> and <P*_y> are redundant and replaced by a transverse polarization:
    histosRing.add("RawPolarization/PolMaps/Aee/p2dPtStar_vsPxAeePyAee", "<P_{#Lambda}^{*}>_{T} vs (p_{x,AEE}^{#Lambda},p_{y,AEE}^{#Lambda});p_{x,AEE}^{#Lambda} (GeV/c);p_{y,AEE}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{T}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
    histosRing.add("RawPolarization/PolMaps/Aee/p2dPzStar_vsPxAeePyAee", "<P_{#Lambda}^{*}>_{z} vs (p_{x,AEE}^{#Lambda},p_{y,AEE}^{#Lambda});p_{x,AEE}^{#Lambda} (GeV/c);p_{y,AEE}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{z}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
    histosRing.add("RawPolarization/PolMaps/Aee/h2dCountsVsPxAeePyAee", "V0 Counts vs (p_{x,AEE}^{#Lambda},p_{y,AEE}^{#Lambda});p_{x,AEE}^{#Lambda} (GeV/c);p_{y,AEE}^{#Lambda} (GeV/c);Counts", kTH2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});

    histosRing.add("RawPolarization/PolMaps/Aee/p2dPtStar_vsPzPxAee", "<P_{#Lambda}^{*}>_{T} vs (p_{z}^{#Lambda},p_{x,AEE}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{x,AEE}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{T}", kTProfile2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPRot});
    histosRing.add("RawPolarization/PolMaps/Aee/p2dPzStar_vsPzPxAee", "<P_{#Lambda}^{*}>_{z} vs (p_{z}^{#Lambda},p_{x,AEE}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{x,AEE}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{z}", kTProfile2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPRot});
    histosRing.add("RawPolarization/PolMaps/Aee/h2dCountsVsPzPxAee", "V0 Counts vs (p_{z}^{#Lambda},p_{x,AEE}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{x,AEE}^{#Lambda} (GeV/c);Counts", kTH2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPRot});

    histosRing.add("RawPolarization/PolMaps/Aee/p2dPtStar_vsPyAeePz", "<P_{#Lambda}^{*}>_{T} vs (p_{y,AEE}^{#Lambda},p_{z}^{#Lambda});p_{y,AEE}^{#Lambda} (GeV/c);p_{z}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{T}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPz});
    histosRing.add("RawPolarization/PolMaps/Aee/p2dPzStar_vsPyAeePz", "<P_{#Lambda}^{*}>_{z} vs (p_{y,AEE}^{#Lambda},p_{z}^{#Lambda});p_{y,AEE}^{#Lambda} (GeV/c);p_{z}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{z}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPz});
    histosRing.add("RawPolarization/PolMaps/Aee/h2dCountsVsPyAeePz", "V0 Counts vs (p_{y,AEE}^{#Lambda},p_{z}^{#Lambda});p_{y,AEE}^{#Lambda} (GeV/c);p_{z}^{#Lambda} (GeV/c);Counts", kTH2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPz});

    // PrimeV0 frame (\hat x'_V0 along the Lambda's transverse direction):
    // (<P*_y'V0> is the ring observable computed with the beam as the jet proxy)
    histosRing.add("RawPolarization/PolMaps/PrimeV0/p2dPxStarPrimeV0_vsPzPt", "<P_{#Lambda}^{*}>_{x'V0} vs (p_{z}^{#Lambda},p_{T}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{T}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{x'V0}", kTProfile2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPtRot});
    histosRing.add("RawPolarization/PolMaps/PrimeV0/p2dPyStarPrimeV0_vsPzPt", "<P_{#Lambda}^{*}>_{y'V0} vs (p_{z}^{#Lambda},p_{T}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{T}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{y'V0}", kTProfile2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPtRot});
    histosRing.add("RawPolarization/PolMaps/PrimeV0/p2dPzStar_vsPzPt", "<P_{#Lambda}^{*}>_{z} vs (p_{z}^{#Lambda},p_{T}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{T}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{z}", kTProfile2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPtRot});
    histosRing.add("RawPolarization/PolMaps/PrimeV0/h2dCountsVsPzPt", "V0 Counts vs (p_{z}^{#Lambda},p_{T}^{#Lambda});p_{z}^{#Lambda} (GeV/c);p_{T}^{#Lambda} (GeV/c);Counts", kTH2D, {axisConfigurations.axisLambdaPz, axisConfigurations.axisLambdaPtRot});

    // PrimeJet frame (\hat z_Jet = jet, \hat x_Jet = beam direction, orthogonalised against the jet)
    histosRing.add("RawPolarization/PolMaps/PrimeJet/p2dPxStarPrimeJet_vsPxPyPrimeJet", "<P_{#Lambda}^{*}>_{x'Jet} vs (p_{x'Jet}^{#Lambda},p_{y'Jet}^{#Lambda});p_{x'Jet}^{#Lambda} (GeV/c);p_{y'Jet}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{x'Jet}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
    histosRing.add("RawPolarization/PolMaps/PrimeJet/p2dPyStarPrimeJet_vsPxPyPrimeJet", "<P_{#Lambda}^{*}>_{y'Jet} vs (p_{x'Jet}^{#Lambda},p_{y'Jet}^{#Lambda});p_{x'Jet}^{#Lambda} (GeV/c);p_{y'Jet}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{y'Jet}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
    histosRing.add("RawPolarization/PolMaps/PrimeJet/p2dPzStarPrimeJet_vsPxPyPrimeJet", "<P_{#Lambda}^{*}>_{z'Jet} vs (p_{x'Jet}^{#Lambda},p_{y'Jet}^{#Lambda});p_{x'Jet}^{#Lambda} (GeV/c);p_{y'Jet}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{z'Jet}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
    histosRing.add("RawPolarization/PolMaps/PrimeJet/h2dCountsVsPxPyPrimeJet", "V0 Counts vs (p_{x'Jet}^{#Lambda},p_{y'Jet}^{#Lambda});p_{x'Jet}^{#Lambda} (GeV/c);p_{y'Jet}^{#Lambda} (GeV/c);Counts", kTH2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});

    histosRing.add("RawPolarization/PolMaps/PrimeJet/p2dPxStarPrimeJet_vsPzPxPrimeJet", "<P_{#Lambda}^{*}>_{x'Jet} vs (p_{z'Jet}^{#Lambda},p_{x'Jet}^{#Lambda});p_{z'Jet}^{#Lambda} (GeV/c);p_{x'Jet}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{x'Jet}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
    histosRing.add("RawPolarization/PolMaps/PrimeJet/p2dPyStarPrimeJet_vsPzPxPrimeJet", "<P_{#Lambda}^{*}>_{y'Jet} vs (p_{z'Jet}^{#Lambda},p_{x'Jet}^{#Lambda});p_{z'Jet}^{#Lambda} (GeV/c);p_{x'Jet}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{y'Jet}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
    histosRing.add("RawPolarization/PolMaps/PrimeJet/p2dPzStarPrimeJet_vsPzPxPrimeJet", "<P_{#Lambda}^{*}>_{z'Jet} vs (p_{z'Jet}^{#Lambda},p_{x'Jet}^{#Lambda});p_{z'Jet}^{#Lambda} (GeV/c);p_{x'Jet}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{z'Jet}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
    histosRing.add("RawPolarization/PolMaps/PrimeJet/h2dCountsVsPzPxPrimeJet", "V0 Counts vs (p_{z'Jet}^{#Lambda},p_{x'Jet}^{#Lambda});p_{z'Jet}^{#Lambda} (GeV/c);p_{x'Jet}^{#Lambda} (GeV/c);Counts", kTH2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});

    histosRing.add("RawPolarization/PolMaps/PrimeJet/p2dPxStarPrimeJet_vsPyPzPrimeJet", "<P_{#Lambda}^{*}>_{x'Jet} vs (p_{y'Jet}^{#Lambda},p_{z'Jet}^{#Lambda});p_{y'Jet}^{#Lambda} (GeV/c);p_{z'Jet}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{x'Jet}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
    histosRing.add("RawPolarization/PolMaps/PrimeJet/p2dPyStarPrimeJet_vsPyPzPrimeJet", "<P_{#Lambda}^{*}>_{y'Jet} vs (p_{y'Jet}^{#Lambda},p_{z'Jet}^{#Lambda});p_{y'Jet}^{#Lambda} (GeV/c);p_{z'Jet}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{y'Jet}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
    histosRing.add("RawPolarization/PolMaps/PrimeJet/p2dPzStarPrimeJet_vsPyPzPrimeJet", "<P_{#Lambda}^{*}>_{z'Jet} vs (p_{y'Jet}^{#Lambda},p_{z'Jet}^{#Lambda});p_{y'Jet}^{#Lambda} (GeV/c);p_{z'Jet}^{#Lambda} (GeV/c);<P_{#Lambda}^{*}>_{z'Jet}", kTProfile2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});
    histosRing.add("RawPolarization/PolMaps/PrimeJet/h2dCountsVsPyPzPrimeJet", "V0 Counts vs (p_{y'Jet}^{#Lambda},p_{z'Jet}^{#Lambda});p_{y'Jet}^{#Lambda} (GeV/c);p_{z'Jet}^{#Lambda} (GeV/c);Counts", kTH2D, {axisConfigurations.axisLambdaPRot, axisConfigurations.axisLambdaPRot});

    // ===============================
    // QA histograms - Useful numbers
    // ===============================
    // (TODO: implement these!)
    // (TODO: implement momentum imbalance checks for jets!)
    // Added to a separate folder for further control:
    // histosRing.add("QA_Numbers/Ring/hValidLeadJets", "hValidLeadJets", kTH1D, {{1,0,1}});
    // TODO: Add "frequency of jets per pT" histograms either here or in the TableProducer

    // Estimating error bars with the Delta Method for <R> = A/B:
    // 1D Delta Method for Integrated observable:
    histosRing.add("Ring/DeltaMethod/hIntegrated", "Delta Method Accumulators Integrated;Component;Counts", kTH1D, {axisConfigurations.axisDeltaComponents});

    // 2D Delta Method for Differentials
    histosRing.add("Ring/DeltaMethod/h2dDeltaThetaVsDeltaComp", "Delta Method vs #Delta#theta_{jet};#Delta#theta_{jet};Component", kTH2D, {axisConfigurations.axisDeltaTheta, axisConfigurations.axisDeltaComponents});
    histosRing.add("Ring/DeltaMethod/h2dLambdaPtVsDeltaComp", "Delta Method vs #Lambda #it{p}_{T};#it{p}_{T}^{#Lambda} (GeV/c);Component", kTH2D, {axisConfigurations.axisPt, axisConfigurations.axisDeltaComponents});
    histosRing.add("Ring/DeltaMethod/h2dMassVsDeltaComp", "Delta Method vs Mass;m_{p#pi} (GeV/c^{2});Component", kTH2D, {axisConfigurations.axisLambdaMassSigExtract, axisConfigurations.axisDeltaComponents});

    // ===============================
    // SelectionFlow/
    // ===============================
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
 
      auto hAnalysisLevelSelectionV0s = histos.add<TH1>("SelectionFlow/hAnalysisLevelSelectionV0s", "Analysis-level V0 selection flow", kTH1D, {{nAnalysisCutBins, -0.5, static_cast<double>(nAnalysisCutBins) - 0.5}});
      // Same flow against the Lambda-like mass, to tell background rejection apart from plain V0 candidate loss (same GeneralQA/h2dSelectionLambdaMass):
      auto h2dAnalysisLevelSelectionV0sVsMass = histos.add<TH2>("SelectionFlow/h2dAnalysisLevelSelectionV0sVsMass", "Analysis-level V0 selection flow vs M_{#Lambda-like}; ;m_{p#pi} (GeV/c^{2})", kTH2D, {{nAnalysisCutBins, -0.5, static_cast<double>(nAnalysisCutBins) - 0.5}, axisConfigurations.axisLambdaMassSigExtract});
      for (int i = 0; i < nAnalysisCutBins; ++i) {
        auto lbl = v0AnalysisCutLabels[i].label;
        if (!v0AnalysisCutLabels[i].enabled)
          lbl = "#color[16]{(off) " + lbl + "}";
        hAnalysisLevelSelectionV0s->GetXaxis()->SetBinLabel(i + 1, lbl.c_str()); // First non-underflow bin is bin 1
        h2dAnalysisLevelSelectionV0sVsMass->GetXaxis()->SetBinLabel(i + 1, lbl.c_str());
      }
      histos.add("SelectionFlow/h2dArmenterosInput", "h2dArmenterosInput;Armenteros #alpha;Armenteros q_{T} (GeV/c)", kTH2D, {axisConfigurations.axisAPAlpha, axisConfigurations.axisAPQt});
      histos.add("SelectionFlow/h2dArmenterosAnalysisCuts", "h2dArmenterosAnalysisCuts;Armenteros #alpha;Armenteros q_{T} (GeV/c)", kTH2D, {axisConfigurations.axisAPAlpha, axisConfigurations.axisAPQt});
    }

    // Kinematic QA histograms for diagnosing detector defficiencies (V0s and daughters here. The proxies are under KinematicsQA/<Proxy>/, booked further below):
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
    histos.get<TProfile>(HIST("IntegratedCuts/SubJet/pRingCuts"))->GetXaxis()->SetBinLabel(2, "p_{T}^{#Lambda}@[0.5,1.5],|y_{#Lambda}|<0.5");
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
    if (doResamplingPassQA) {
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
    histos.get<TProfile2D>(HIST("IntegratedCuts/SubJet/p2dRingCutsV0MassPeak"))->GetXaxis()->SetBinLabel(2, "p_{T}^{#Lambda}@[0.5,1.5],|y_{#Lambda}|<0.5");
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
    histos.get<TH1>(HIST("IntegratedCuts/SubJet/hCountCuts"))->GetXaxis()->SetBinLabel(2, "p_{T}^{#Lambda}@[0.5,1.5],|y_{#Lambda}|<0.5");
    histos.get<TH1>(HIST("IntegratedCuts/SubJet/hCountCuts"))->GetXaxis()->SetBinLabel(3, "|SubJet_{#eta}|<0.5");
    histos.get<TH1>(HIST("IntegratedCuts/SubJet/hCountCuts"))->GetXaxis()->SetBinLabel(4, "#Lambda + SubJet cuts");

    histos.add("IntegratedCuts/LeadP/hCountCuts", "hCountCuts; ;N V0s", kTH1D, {{4, 0, 4}});
    histos.get<TH1>(HIST("IntegratedCuts/LeadP/hCountCuts"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
    histos.get<TH1>(HIST("IntegratedCuts/LeadP/hCountCuts"))->GetXaxis()->SetBinLabel(2, "p_{T}^{#Lambda}@[0.5,1.5],|y_{#Lambda}|<0.5");
    histos.get<TH1>(HIST("IntegratedCuts/LeadP/hCountCuts"))->GetXaxis()->SetBinLabel(3, "|LeadP_{#eta}|<0.5");
    histos.get<TH1>(HIST("IntegratedCuts/LeadP/hCountCuts"))->GetXaxis()->SetBinLabel(4, "#Lambda + LeadP cuts");

    // Fake-polarization diagnostics:
    if (qaSwitches.doFakePolDiagnosticsQA) {
      // ===============================
      // FakePolDrivers/: V0-related and Proxy-related dependencies that greatly influence the Full 3D ring signal
      // ===============================
      // Studying the magnetic field dependence of particle reconstruction efficiency (not magnitude, just sign of field):
      // (species-agnostic: the population is set by analyseLambda/analyseAntiLambda, so the species split lives in separate configs)
      if (analyseMagField) {
        histos.add("FakePolDrivers/hMassDecayGeomRight", "hMassDecayGeomRight; m_{Inv}; Counts", kTH1D, {axisConfigurations.axisLambdaMass});
        histos.add("FakePolDrivers/hMassDecayGeomLeft", "hMassDecayGeomLeft; m_{Inv}; Counts", kTH1D, {axisConfigurations.axisLambdaMass});
      }

      // ===============================
      // FakePolDrivers/CosHEE/: helicity efficiency effect
      // ===============================
      // Fake polarization signal QA
      // --> The "negative helicity problem", where topologies with a proton decaying opposite to the Lambda momentum are enhanced by
      // efficiency of reconstruction. The geometries where the proton moves in the same direction as the boost will have a very small
      // momentum pion, which is not as easily detected as the opposite case! This may insert a fake signal of polarization in the measurement!
      histos.add("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCounts", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda};", kTH2D, {axisConfigurations.axisCosTheta, {9, 0, 9}});
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCounts"))->GetZaxis()->SetTitle("N_{V0s}");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(2, "#eta_{Jet} #geq 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(3, "#eta_{Jet} < 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(6, "#eta_{Jet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(7, "#eta_{Jet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(8, "#eta_{Jet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(9, "#eta_{Jet} < 0, #eta_{#Lambda} < 0");

      // The same, but for actual signal instead of counts:
      histos.add("FakePolDrivers/CosHEE/LeadJet/p2dFakePolSignalVsCosTheta", "FakePolSignal; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda};", kTProfile2D, {axisConfigurations.axisCosTheta, {9, 0, 9}});
      histos.get<TProfile2D>(HIST("FakePolDrivers/CosHEE/LeadJet/p2dFakePolSignalVsCosTheta"))->GetZaxis()->SetTitle("<#it{R}>");
      histos.get<TProfile2D>(HIST("FakePolDrivers/CosHEE/LeadJet/p2dFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TProfile2D>(HIST("FakePolDrivers/CosHEE/LeadJet/p2dFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(2, "#eta_{Jet} #geq 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/CosHEE/LeadJet/p2dFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(3, "#eta_{Jet} < 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/CosHEE/LeadJet/p2dFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/CosHEE/LeadJet/p2dFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/CosHEE/LeadJet/p2dFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(6, "#eta_{Jet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/CosHEE/LeadJet/p2dFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(7, "#eta_{Jet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/CosHEE/LeadJet/p2dFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(8, "#eta_{Jet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/CosHEE/LeadJet/p2dFakePolSignalVsCosTheta"))->GetYaxis()->SetBinLabel(9, "#eta_{Jet} < 0, #eta_{#Lambda} < 0");

      // Doing the same HEE study for leading particles:
      // (eta_{Jet} may be a bad estimator!)
      histos.add("FakePolDrivers/CosHEE/LeadP/h2dFakePolCounts", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda};", kTH2D, {axisConfigurations.axisCosTheta, {9, 0, 9}});
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadP/h2dFakePolCounts"))->GetZaxis()->SetTitle("N_{V0s}");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadP/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadP/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(2, "#eta_{LeadP} #geq 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadP/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(3, "#eta_{LeadP} < 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadP/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadP/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadP/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(6, "#eta_{LeadP} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadP/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(7, "#eta_{LeadP} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadP/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(8, "#eta_{LeadP} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadP/h2dFakePolCounts"))->GetYaxis()->SetBinLabel(9, "#eta_{LeadP} < 0, #eta_{#Lambda} < 0");

      // Avoid fake signal by jets boosting the Lambda in its own direction, then modifying efficiency of reconstruction in a similar way:
      histos.add("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsLambdaPtCut", "FakePol,p_{T}^{#Lambda}#in[0.5,1.5]; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda};", kTH2D, {axisConfigurations.axisCosTheta, {9, 0, 9}});
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsLambdaPtCut"))->GetZaxis()->SetTitle("N_{V0s}");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(2, "#eta_{Jet} #geq 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(3, "#eta_{Jet} < 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(6, "#eta_{Jet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(7, "#eta_{Jet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(8, "#eta_{Jet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsLambdaPtCut"))->GetYaxis()->SetBinLabel(9, "#eta_{Jet} < 0, #eta_{#Lambda} < 0");

      // Even stricter cut (also demands rapidity cut stricter than jets, so may see different boosting):
      histos.add("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsLambdaPtYCuts", "FakePol,p_{T}^{#Lambda}#in[0.5,1.5],|y_{#Lambda}|<0.5; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda};", kTH2D, {axisConfigurations.axisCosTheta, {9, 0, 9}});
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsLambdaPtYCuts"))->GetZaxis()->SetTitle("N_{V0s}");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(2, "#eta_{Jet} #geq 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(3, "#eta_{Jet} < 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(6, "#eta_{Jet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(7, "#eta_{Jet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(8, "#eta_{Jet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsLambdaPtYCuts"))->GetYaxis()->SetBinLabel(9, "#eta_{Jet} < 0, #eta_{#Lambda} < 0");

      // Another useful quantity -- How much is the fake signal related to the jet's momentum (how much the fake signal is correlated with the Jet-Lambda angular separation):
      histos.add("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsVsDeltaTheta", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #Delta#theta_{Jet}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaTheta});
      histos.add("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsVsDeltaThetaPosProxyEta", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #Delta#theta_{Jet}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaTheta});
      histos.add("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsVsDeltaThetaNegProxyEta", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #Delta#theta_{Jet}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaTheta});
      histos.add("FakePolDrivers/CosHEE/LeadP/h2dFakePolCountsVsDeltaTheta", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #Delta#theta_{LeadP}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaTheta});
      histos.add("FakePolDrivers/CosHEE/LeadP/h2dFakePolCountsVsDeltaThetaPosProxyEta", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #Delta#theta_{LeadP}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaTheta});
      histos.add("FakePolDrivers/CosHEE/LeadP/h2dFakePolCountsVsDeltaThetaNegProxyEta", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #Delta#theta_{LeadP}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaTheta});

      // Understanding the dip at the cos = -1 end:
      histos.add("FakePolDrivers/CosHEE/LeadJet/h2dFakePolCountsCosThetaVsLambdaPt", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; p_{T}^{#Lambda}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisPtCoarseQA});
      histos.add("FakePolDrivers/CosHEE/LeadP/h2dFakePolCountsCosThetaVsLambdaPt", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; p_{T}^{#Lambda}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisPtCoarseQA});

      // The Helicity Efficiency Effect resolved in the five mass regions (left sideband, left gap, peak, right gap, right sideband).
      histos.add("FakePolDrivers/CosHEE/ByMass/h2dCountsCosThetaHEEVsFullMass", "Counts vs cos(#theta)_{HEE} vs mass, all V0s;cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda-like};m_{p#pi} (GeV/c^{2});<#it{R}>", kTH2D, {axisConfigurations.axisCosThetaCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("FakePolDrivers/CosHEE/ByMass/p2dRingCosThetaHEEVsFullMass", "<#it{R}>_{LeadJet} vs cos(#theta)_{HEE} vs mass slice, all V0s;cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda-like};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisCosThetaCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("FakePolDrivers/CosHEE/ByMass/p2dRingCosThetaHEEVsMass", "<#it{R}>_{LeadJet} vs cos(#theta)_{HEE} vs mass slice, all V0s;cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda-like};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisCosThetaCoarse, axisMassRegions});
      histos.add("FakePolDrivers/CosHEE/ByMass/p2dRingCosThetaHEEVsMassPosPhiAEE", "<#it{R}>_{LeadJet} (#phi_{#Lambda}-#phi_{p}^{*}>0) vs cos(#theta)_{HEE} vs mass slice, all V0s;cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda-like};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisCosThetaCoarse, axisMassRegions});
      histos.add("FakePolDrivers/CosHEE/ByMass/p2dRingCosThetaHEEVsMassNegPhiAEE", "<#it{R}>_{LeadJet} (#phi_{#Lambda}-#phi_{p}^{*}<0) vs cos(#theta)_{HEE} vs mass slice, all V0s;cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda-like};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisCosThetaCoarse, axisMassRegions});

      // ===============================
      // FakePolDrivers/PhiAEE/: azimuthal efficiency effect
      // ===============================
      // For the phi_Lambda - phi_D^* dependency as well:
      histos.add("FakePolDrivers/PhiAEE/LeadJet/p2dFakePolSignalVsPhiAEE", "FakePolSignal, AEE dependence; #phi_{#Lambda}-#phi_{p}^{*};", kTProfile2D, {axisConfigurations.axisDeltaPhi, {9, 0, 9}});
      histos.get<TProfile2D>(HIST("FakePolDrivers/PhiAEE/LeadJet/p2dFakePolSignalVsPhiAEE"))->GetZaxis()->SetTitle("<#it{R}>");
      histos.get<TProfile2D>(HIST("FakePolDrivers/PhiAEE/LeadJet/p2dFakePolSignalVsPhiAEE"))->GetYaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TProfile2D>(HIST("FakePolDrivers/PhiAEE/LeadJet/p2dFakePolSignalVsPhiAEE"))->GetYaxis()->SetBinLabel(2, "#eta_{Jet} #geq 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/PhiAEE/LeadJet/p2dFakePolSignalVsPhiAEE"))->GetYaxis()->SetBinLabel(3, "#eta_{Jet} < 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/PhiAEE/LeadJet/p2dFakePolSignalVsPhiAEE"))->GetYaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/PhiAEE/LeadJet/p2dFakePolSignalVsPhiAEE"))->GetYaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/PhiAEE/LeadJet/p2dFakePolSignalVsPhiAEE"))->GetYaxis()->SetBinLabel(6, "#eta_{Jet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/PhiAEE/LeadJet/p2dFakePolSignalVsPhiAEE"))->GetYaxis()->SetBinLabel(7, "#eta_{Jet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/PhiAEE/LeadJet/p2dFakePolSignalVsPhiAEE"))->GetYaxis()->SetBinLabel(8, "#eta_{Jet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/PhiAEE/LeadJet/p2dFakePolSignalVsPhiAEE"))->GetYaxis()->SetBinLabel(9, "#eta_{Jet} < 0, #eta_{#Lambda} < 0");

      // Also including an observable that probes spectrum broadening due to the "Azimuthal Efficiency Effect":
      histos.add("FakePolDrivers/PhiAEE/h2dMassVsPhiAEE", "m_{#Lambda-like}, AEE probe; m_{Inv}; #phi_{#Lambda-like}-#phi_{p-like}^{*} ; Counts", kTH2D, {axisConfigurations.axisLambdaMass, axisConfigurations.axisDeltaPhi});

      // <R> Vs PhiLambda-PhiProtonStar dependencies:
      // (species-agnostic: the population is set by analyseLambda/analyseAntiLambda, so the species split lives in separate configs)
      histos.add("FakePolDrivers/PhiAEE/LeadJet/pRingObservableVsPhiAEE", "<#it{R}>_{LeadJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*};#phi_{#Lambda-like}-#phi_{p-like}^{*};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("FakePolDrivers/PhiAEE/LeadP/pRingObservableVsPhiAEE", "<#it{R}>_{LeadP} vs #phi_{#Lambda-like}-#phi_{p-like}^{*};#phi_{#Lambda-like}-#phi_{p-like}^{*};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("FakePolDrivers/PhiAEE/SubJet/pRingObservableVsPhiAEE", "<#it{R}>_{SubJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*};#phi_{#Lambda-like}-#phi_{p-like}^{*};<#it{R}>", kTProfile, {axisConfigurations.axisDeltaPhi});

      // 2D dependencies for signal extraction:
      // (coarser angular axis here: the mass axis takes the statistics, and each cell still has to sustain a fit)
      histos.add("FakePolDrivers/PhiAEE/LeadJet/p2dRingObservableVsPhiAEEVsMass", "<#it{R}>_{LeadJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("FakePolDrivers/PhiAEE/LeadP/p2dRingObservableVsPhiAEEVsMass", "<#it{R}>_{LeadP} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("FakePolDrivers/PhiAEE/SubJet/p2dRingObservableVsPhiAEEVsMass", "<#it{R}>_{SubJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      // Splitting in proxy eta:
      histos.add("FakePolDrivers/PhiAEE/LeadJet/p2dRingObservableVsPhiAEEVsMassPosProxyEta", "<#it{R}>_{LeadJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass, #eta_{Proxy}>0;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("FakePolDrivers/PhiAEE/LeadP/p2dRingObservableVsPhiAEEVsMassPosProxyEta", "<#it{R}>_{LeadP} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass, #eta_{Proxy}>0;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("FakePolDrivers/PhiAEE/SubJet/p2dRingObservableVsPhiAEEVsMassPosProxyEta", "<#it{R}>_{SubJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass, #eta_{Proxy}>0;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("FakePolDrivers/PhiAEE/LeadJet/p2dRingObservableVsPhiAEEVsMassNegProxyEta", "<#it{R}>_{LeadJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass, #eta_{Proxy}<0;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("FakePolDrivers/PhiAEE/LeadP/p2dRingObservableVsPhiAEEVsMassNegProxyEta", "<#it{R}>_{LeadP} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass, #eta_{Proxy}<0;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("FakePolDrivers/PhiAEE/SubJet/p2dRingObservableVsPhiAEEVsMassNegProxyEta", "<#it{R}>_{SubJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs Mass, #eta_{Proxy}<0;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      // AEE angle Vs Proxy eta dependence:
      histos.add("FakePolDrivers/PhiAEE/LeadJet/p2dRingObservableVsPhiAEEVsProxyEtaSign", "<#it{R}>_{LeadJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy} sign;<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, {2, -1, 1}});
      histos.add("FakePolDrivers/PhiAEE/LeadP/p2dRingObservableVsPhiAEEVsProxyEtaSign", "<#it{R}>_{LeadP} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy} sign;<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, {2, -1, 1}});
      histos.add("FakePolDrivers/PhiAEE/SubJet/p2dRingObservableVsPhiAEEVsProxyEtaSign", "<#it{R}>_{SubJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy} sign;<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, {2, -1, 1}});
      histos.add("FakePolDrivers/PhiAEE/LeadJet/p2dRingObservableVsPhiAEEVsProxyEta", "<#it{R}>_{LeadJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy};<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEta});
      histos.add("FakePolDrivers/PhiAEE/LeadP/p2dRingObservableVsPhiAEEVsProxyEta", "<#it{R}>_{LeadP} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy};<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEta});
      histos.add("FakePolDrivers/PhiAEE/SubJet/p2dRingObservableVsPhiAEEVsProxyEta", "<#it{R}>_{SubJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy};<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEta});
      // A 3D split that allows for a proper signal extraction after separating the main drivers of fake polarization signal:
      histos.add("FakePolDrivers/PhiAEE/LeadJet/p3dRingObservableVsPhiAEEVsProxyEtaVsMass", "<#it{R}>_{LeadJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy}; m_{p#pi};<#it{R}>", kTProfile3D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("FakePolDrivers/PhiAEE/LeadP/p3dRingObservableVsPhiAEEVsProxyEtaVsMass", "<#it{R}>_{LeadP} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy}; m_{p#pi};<#it{R}>", kTProfile3D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("FakePolDrivers/PhiAEE/SubJet/p3dRingObservableVsPhiAEEVsProxyEtaVsMass", "<#it{R}>_{SubJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs #eta_{Proxy};#phi_{#Lambda-like}-#phi_{p-like}^{*}; #eta_{Proxy}; m_{p#pi};<#it{R}>", kTProfile3D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisEtaSigExtract, axisConfigurations.axisLambdaMassSigExtract});

      // The same profiles collapsed onto the five mass regions:
      // (high statistics for a simpler lookup)
      histos.add("FakePolDrivers/PhiAEE/FiveBinMass/LeadJet/p2dRingObservableVsPhiAEEVs5BinMass", "<#it{R}>_{LeadJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs mass slice;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisMassRegions});
      histos.add("FakePolDrivers/PhiAEE/FiveBinMass/LeadP/p2dRingObservableVsPhiAEEVs5BinMass", "<#it{R}>_{LeadP} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs mass slice;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisMassRegions});
      histos.add("FakePolDrivers/PhiAEE/FiveBinMass/SubJet/p2dRingObservableVsPhiAEEVs5BinMass", "<#it{R}>_{SubJet} vs #phi_{#Lambda-like}-#phi_{p-like}^{*} Vs mass slice;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisMassRegions});

      // Picking a representative split in eta and PhiAEE just to see if we manage to stabilize the background (based off p2dRingObservableEtaLambdaVsMass, but booked outside the Ring/ folder):
      histos.add("FakePolDrivers/PhiAEE/BkgStabilizationAttempt/p2dRingObservableEtaLambdaVsMassEtaProxyPosPhiAEEPos", "<#it{R}> vs #eta_{#Lambda} vs Mass, #eta_{Proxy}>0, #phi_{AEE}>0;#eta_{#Lambda};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("FakePolDrivers/PhiAEE/BkgStabilizationAttempt/p2dRingObservableEtaLambdaVsMassEtaProxyNegPhiAEEPos", "<#it{R}> vs #eta_{#Lambda} vs Mass, #eta_{Proxy}<0, #phi_{AEE}>0;#eta_{#Lambda};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("FakePolDrivers/PhiAEE/BkgStabilizationAttempt/p2dRingObservableEtaLambdaVsMassEtaProxyPosPhiAEEPosDeltaPhiLeadJetPos", "<#it{R}> vs #eta_{#Lambda} vs Mass, #eta_{Proxy}>0, #phi_{AEE}>0, #Delta#phi_{Jet}>0;#eta_{#Lambda};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("FakePolDrivers/PhiAEE/BkgStabilizationAttempt/p2dRingObservableEtaLambdaVsMassEtaProxyPosPhiAEEPosDeltaPhiLeadJetNeg", "<#it{R}> vs #eta_{#Lambda} vs Mass, #eta_{Proxy}>0, #phi_{AEE}>0, #Delta#phi_{Jet}<0;#eta_{#Lambda};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisV0EtaCoarse, axisConfigurations.axisLambdaMassSigExtract});
      // Splitting in the jet coordinate, but leaving PhiAEE differential:
      histos.add("FakePolDrivers/PhiAEE/BkgStabilizationAttempt/p2dRingObservablePhiAEEVsMassEtaProxyPosPhiEtaLambdaPos", "<#it{R}> vs #phi_{#Lambda-like}-#phi_{p-like}^{*} vs Mass, #eta_{Proxy}>0, #eta_{#Lambda}>0;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("FakePolDrivers/PhiAEE/BkgStabilizationAttempt/p2dRingObservablePhiAEEVsMassEtaProxyNegPhiEtaLambdaPos", "<#it{R}> vs #phi_{#Lambda-like}-#phi_{p-like}^{*} vs Mass, #eta_{Proxy}<0, #eta_{#Lambda}>0;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("FakePolDrivers/PhiAEE/BkgStabilizationAttempt/p2dRingObservablePhiAEEVsMassEtaProxyPosPhiEtaLambdaPosDeltaPhiLeadJetPos", "<#it{R}> vs #phi_{#Lambda-like}-#phi_{p-like}^{*} vs Mass, #eta_{Proxy}>0, #eta_{#Lambda}>0, #Delta#phi_{Jet}>0;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("FakePolDrivers/PhiAEE/BkgStabilizationAttempt/p2dRingObservablePhiAEEVsMassEtaProxyPosPhiEtaLambdaPosDeltaPhiLeadJetNeg", "<#it{R}> vs #phi_{#Lambda-like}-#phi_{p-like}^{*} vs Mass, #eta_{Proxy}>0, #eta_{#Lambda}>0, #Delta#phi_{Jet}<0;#phi_{#Lambda-like}-#phi_{p-like}^{*};m_{p#pi} (GeV/c^{2});<#it{R}>", kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});

      // ===============================
      // FakePolDrivers/PhiStar/: decay-plane azimuth phi*
      // ===============================
      // Similar split, but for AEE instead of HEE:
      histos.add("FakePolDrivers/PhiStar/LeadJet/h2dCountsVsPhiStar", "FakePolCounts, AEE dependence; #phi^{*};", kTH2D, {axisConfigurations.axisDeltaPhi, {9, 0, 9}});
      histos.get<TH2>(HIST("FakePolDrivers/PhiStar/LeadJet/h2dCountsVsPhiStar"))->GetZaxis()->SetTitle("Counts");
      histos.get<TH2>(HIST("FakePolDrivers/PhiStar/LeadJet/h2dCountsVsPhiStar"))->GetYaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TH2>(HIST("FakePolDrivers/PhiStar/LeadJet/h2dCountsVsPhiStar"))->GetYaxis()->SetBinLabel(2, "#eta_{Jet} #geq 0");
      histos.get<TH2>(HIST("FakePolDrivers/PhiStar/LeadJet/h2dCountsVsPhiStar"))->GetYaxis()->SetBinLabel(3, "#eta_{Jet} < 0");
      histos.get<TH2>(HIST("FakePolDrivers/PhiStar/LeadJet/h2dCountsVsPhiStar"))->GetYaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("FakePolDrivers/PhiStar/LeadJet/h2dCountsVsPhiStar"))->GetYaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("FakePolDrivers/PhiStar/LeadJet/h2dCountsVsPhiStar"))->GetYaxis()->SetBinLabel(6, "#eta_{Jet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("FakePolDrivers/PhiStar/LeadJet/h2dCountsVsPhiStar"))->GetYaxis()->SetBinLabel(7, "#eta_{Jet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TH2>(HIST("FakePolDrivers/PhiStar/LeadJet/h2dCountsVsPhiStar"))->GetYaxis()->SetBinLabel(8, "#eta_{Jet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TH2>(HIST("FakePolDrivers/PhiStar/LeadJet/h2dCountsVsPhiStar"))->GetYaxis()->SetBinLabel(9, "#eta_{Jet} < 0, #eta_{#Lambda} < 0");
      // For the ring observable as well:
      // Explicitly, checking the Phi* dependence on a series of #eta cuts.
      // The fake, AEE-induced, signal should be zero when integrating on full solid angle, and then have some shape for each eta slice.
      histos.add("FakePolDrivers/PhiStar/LeadJet/p2dFakePolSignalVsPhiStar", "FakePolSignal, AEE dependence; #phi^{*};", kTProfile2D, {axisConfigurations.axisDeltaPhi, {9, 0, 9}});
      histos.get<TProfile2D>(HIST("FakePolDrivers/PhiStar/LeadJet/p2dFakePolSignalVsPhiStar"))->GetZaxis()->SetTitle("<#it{R}>");
      histos.get<TProfile2D>(HIST("FakePolDrivers/PhiStar/LeadJet/p2dFakePolSignalVsPhiStar"))->GetYaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TProfile2D>(HIST("FakePolDrivers/PhiStar/LeadJet/p2dFakePolSignalVsPhiStar"))->GetYaxis()->SetBinLabel(2, "#eta_{Jet} #geq 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/PhiStar/LeadJet/p2dFakePolSignalVsPhiStar"))->GetYaxis()->SetBinLabel(3, "#eta_{Jet} < 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/PhiStar/LeadJet/p2dFakePolSignalVsPhiStar"))->GetYaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/PhiStar/LeadJet/p2dFakePolSignalVsPhiStar"))->GetYaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/PhiStar/LeadJet/p2dFakePolSignalVsPhiStar"))->GetYaxis()->SetBinLabel(6, "#eta_{Jet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/PhiStar/LeadJet/p2dFakePolSignalVsPhiStar"))->GetYaxis()->SetBinLabel(7, "#eta_{Jet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/PhiStar/LeadJet/p2dFakePolSignalVsPhiStar"))->GetYaxis()->SetBinLabel(8, "#eta_{Jet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/PhiStar/LeadJet/p2dFakePolSignalVsPhiStar"))->GetYaxis()->SetBinLabel(9, "#eta_{Jet} < 0, #eta_{#Lambda} < 0");

      // Seeing if phi* is indeed influenced by the DCA between daughters:
      histos.add("FakePolDrivers/PhiStar/DCACorrelations/LeadJet/h2dFakePolCounts_PhiStarVsDCAdau", "FakePolCounts - HasValidLeadJet OK; #phi^{*}; DCA_{V0 Daughters}", kTH2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdau});
      histos.add("FakePolDrivers/PhiStar/DCACorrelations/LeadJet/p2dFakePolSignal_PhiStarVsDCAdau", "FakePolSignal; #phi^{*}; DCA_{V0 Daughters}", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdau});

      // Adding a way to check if the jet eta is positive or negative as well
      // (AEE signal could be closer to zero otherwise: phi^* dependency may not make it fall to zero as we are no longer integrating <R> in full solid angle, yet analyzing this other dependency is also important)
      histos.add("FakePolDrivers/PhiStar/DCACorrelations/LeadJet/p3dFakePolSignal_PhiStarVsDCAdauVsProxyEta", "FakePolSignal; #phi^{*}; DCA_{V0 Daughters}; #eta_{Jet} sign", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdau, {2, -0.9, 0.9}});
      histos.add("FakePolDrivers/PhiStar/DCACorrelations/LeadJet/p3dFakePolSignal_PhiStarVsDCAdauVsEtaLambda", "FakePolSignal; #phi^{*}; DCA_{V0 Daughters}; #eta_{#Lambda} sign", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdau, {2, -0.9, 0.9}});

      // Similar checks for dcaPosToPV and dcaNegToPV, which influence AEE the strongest:
      histos.add("FakePolDrivers/PhiStar/DCACorrelations/LeadJet/h2dFakePolCounts_PhiStarVsDCAProLike", "FakePolCounts - HasValidLeadJet OK; #phi^{*}; DCA_{PosPV}", kTH2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdauPV});
      histos.add("FakePolDrivers/PhiStar/DCACorrelations/LeadJet/p2dFakePolSignal_PhiStarVsDCAProLike", "FakePolSignal; #phi^{*}; DCA_{PosPV}", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdauPV});
      histos.add("FakePolDrivers/PhiStar/DCACorrelations/LeadJet/p3dFakePolSignal_PhiStarVsDCAProLikeVsProxyEta", "FakePolSignal; #phi^{*}; DCA_{PosPV}; #eta_{Jet} sign", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdauPV, {2, -0.9, 0.9}});
      histos.add("FakePolDrivers/PhiStar/DCACorrelations/LeadJet/p3dFakePolSignal_PhiStarVsDCAProLikeVsEtaLambda", "FakePolSignal; #phi^{*}; DCA_{PosPV}; #eta_{#Lambda} sign", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdauPV, {2, -0.9, 0.9}});

      histos.add("FakePolDrivers/PhiStar/DCACorrelations/LeadJet/h2dFakePolCounts_PhiStarVsDCAPiLike", "FakePolCounts - HasValidLeadJet OK; #phi^{*}; DCA_{NegPV}", kTH2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdauPV});
      histos.add("FakePolDrivers/PhiStar/DCACorrelations/LeadJet/p2dFakePolSignal_PhiStarVsDCAPiLike", "FakePolSignal; #phi^{*}; DCA_{NegPV}", kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdauPV});
      histos.add("FakePolDrivers/PhiStar/DCACorrelations/LeadJet/p3dFakePolSignal_PhiStarVsDCAPiLikeVsProxyEta", "FakePolSignal; #phi^{*}; DCA_{NegPV}; #eta_{Jet} sign", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdauPV, {2, -0.9, 0.9}});
      histos.add("FakePolDrivers/PhiStar/DCACorrelations/LeadJet/p3dFakePolSignal_PhiStarVsDCAPiLikeVsEtaLambda", "FakePolSignal; #phi^{*}; DCA_{NegPV}; #eta_{#Lambda} sign", kTProfile3D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisDCAdauPV, {2, -0.9, 0.9}});

      // ===============================
      // FakePolDrivers/CosHEEVsPhiStar/: HEE and phi* together
      // ===============================
      // Seeing the dependence between phi* = atan2(p_p_star \cdot (p_Lambda_hat \times (z_hat \cross p_Lambda)), p_p_star \cdot (z_hat \cross p_Lambda))
      // e_z = p_Lambda_hat; // e_x = normalize(z_hat cross p_Lambda); // e_y = e_z cross e_x;
      // phi_star = atan2(p_p_star dot e_y, p_p_star dot e_x);
      histos.add("FakePolDrivers/CosHEEVsPhiStar/h2dFakePolCounts_CosThetaVsPhiStar", "FakePolCounts; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #phi^{*}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaPhi});
      histos.add("FakePolDrivers/CosHEEVsPhiStar/LeadJet/p2dFakePolSignal_CosThetaVsPhiStar", "FakePolSignal; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #phi^{*}", kTProfile2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaPhi});
      // Specific counter for when we have leading jets (relates directly to LeadJet/p2dFakePolSignal_CosThetaVsPhiStar):
      histos.add("FakePolDrivers/CosHEEVsPhiStar/LeadJet/h2dFakePolCounts_CosThetaVsPhiStar", "FakePolCounts - HasValidLeadJet OK; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; #phi^{*}", kTH2D, {axisConfigurations.axisCosTheta, axisConfigurations.axisDeltaPhi});

      // ===============================
      // FakePolDrivers/ProxyEta/: proxy eta scans
      // ===============================
      // Integrated observable dependent on jet proxy #eta to unfold possible asymmetries in detector:
      histos.add("FakePolDrivers/ProxyEta/LeadJet/pRingEtaCuts", "pRingEtaCuts; ;<#it{R}>", kTProfile, {{15, 0, 15}});
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(2, "#eta_{Jet} #geq 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(3, "#eta_{Jet} < 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(6, "#eta_{Jet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(7, "#eta_{Jet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(8, "#eta_{Jet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(9, "#eta_{Jet} < 0, #eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(10, "#eta_{Jet} > R");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(11, "#eta_{Jet} < -R");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(12, "#eta_{Jet} > R, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(13, "#eta_{Jet} > R, #eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(14, "#eta_{Jet} < -R, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/LeadJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(15, "#eta_{Jet} < -R, #eta_{#Lambda} < 0");

      histos.add("FakePolDrivers/ProxyEta/SubJet/pRingEtaCuts", "pRingEtaCuts; ;<#it{R}>", kTProfile, {{15, 0, 15}});
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(2, "#eta_{SubJet} #geq 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(3, "#eta_{SubJet} < 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(6, "#eta_{SubJet} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(7, "#eta_{SubJet} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(8, "#eta_{SubJet} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(9, "#eta_{SubJet} < 0, #eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(10, "#eta_{SubJet} > R");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(11, "#eta_{SubJet} < -R");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(12, "#eta_{SubJet} > R, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(13, "#eta_{SubJet} > R, #eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(14, "#eta_{SubJet} < -R, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/SubJet/pRingEtaCuts"))->GetXaxis()->SetBinLabel(15, "#eta_{SubJet} < -R, #eta_{#Lambda} < 0");

      histos.add("FakePolDrivers/ProxyEta/LeadP/pRingEtaCuts", "pRingEtaCuts; ;<#it{R}>", kTProfile, {{9, 0, 9}});
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/LeadP/pRingEtaCuts"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/LeadP/pRingEtaCuts"))->GetXaxis()->SetBinLabel(2, "#eta_{LeadP} #geq 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/LeadP/pRingEtaCuts"))->GetXaxis()->SetBinLabel(3, "#eta_{LeadP} < 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/LeadP/pRingEtaCuts"))->GetXaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/LeadP/pRingEtaCuts"))->GetXaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/LeadP/pRingEtaCuts"))->GetXaxis()->SetBinLabel(6, "#eta_{LeadP} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/LeadP/pRingEtaCuts"))->GetXaxis()->SetBinLabel(7, "#eta_{LeadP} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/LeadP/pRingEtaCuts"))->GetXaxis()->SetBinLabel(8, "#eta_{LeadP} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile>(HIST("FakePolDrivers/ProxyEta/LeadP/pRingEtaCuts"))->GetXaxis()->SetBinLabel(9, "#eta_{LeadP} < 0, #eta_{#Lambda} < 0");

      // Studying the signal Vs background integral (a very naive estimative of the invariant mass peak)
      histos.add("FakePolDrivers/ProxyEta/LeadP/p2dRingEtaCuts_MassSignalVsBackground", "p2dRingEtaCuts_MassSignalVsBackground; ; ;<#it{R}>", kTProfile2D, {{9, 0, 9}, {2, 0, 2}});
      histos.get<TProfile2D>(HIST("FakePolDrivers/ProxyEta/LeadP/p2dRingEtaCuts_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(1, "All #Lambda");
      histos.get<TProfile2D>(HIST("FakePolDrivers/ProxyEta/LeadP/p2dRingEtaCuts_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(2, "#eta_{LeadP} #geq 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/ProxyEta/LeadP/p2dRingEtaCuts_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(3, "#eta_{LeadP} < 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/ProxyEta/LeadP/p2dRingEtaCuts_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(4, "#eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/ProxyEta/LeadP/p2dRingEtaCuts_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(5, "#eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/ProxyEta/LeadP/p2dRingEtaCuts_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(6, "#eta_{LeadP} #geq 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/ProxyEta/LeadP/p2dRingEtaCuts_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(7, "#eta_{LeadP} #geq 0, #eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/ProxyEta/LeadP/p2dRingEtaCuts_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(8, "#eta_{LeadP} < 0, #eta_{#Lambda} #geq 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/ProxyEta/LeadP/p2dRingEtaCuts_MassSignalVsBackground"))->GetXaxis()->SetBinLabel(9, "#eta_{LeadP} < 0, #eta_{#Lambda} < 0");
      histos.get<TProfile2D>(HIST("FakePolDrivers/ProxyEta/LeadP/p2dRingEtaCuts_MassSignalVsBackground"))->GetYaxis()->SetBinLabel(1, "#Lambda out of mass peak");
      histos.get<TProfile2D>(HIST("FakePolDrivers/ProxyEta/LeadP/p2dRingEtaCuts_MassSignalVsBackground"))->GetYaxis()->SetBinLabel(2, "#Lambda in mass peak");

      // ===============================
      // FakePolDrivers/ProxyZ/: proxy direction cosine
      // ===============================
      // More about possible AEE dependencies (should see an invariance with JetEta and <R>/jetZ):
      // TODO: think about these error bars: do they still make sense via regular TProfile's SEM error? These are just a quick check, so wouldn't bother much about it.
      histos.add("FakePolDrivers/ProxyZ/LeadJet/pRingVsProxyZ", "<#it{R}> vs #hat{t}_{z}; #hat{t}_{z}; <#it{R}>", kTProfile, {axisConfigurations.axisProxyZ}); // Numerically stable and can also show the sign flip (essentially the <#it{R}> vs Eta Jet plot in another scale)
      histos.add("FakePolDrivers/ProxyZ/LeadJet/pRingOverProxyZ_VsProxyEta", "<#it{R}>/#hat{t}_{z}; #eta_{Jet}; <#it{R}>/#hat{t}_{z}", kTProfile, {axisConfigurations.axisEtaCoarse});
      histos.add("FakePolDrivers/ProxyZ/LeadJet/pRingOverProxyZ_VsCosThetaHEE", "<#it{R}>/#hat{t}_{z} Vs cos(#theta) HEE; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; <#it{R}>/#hat{t}_{z}", kTProfile, {axisConfigurations.axisCosTheta});
      histos.add("FakePolDrivers/ProxyZ/LeadJet/pRingOverProxyZ_VsPhiStar", "<#it{R}>/#hat{t}_{z} Vs #phi^{*}; #phi^{*} = atan2(#vec{p}^{*}_{p} #upoint [#hat{p}_{#Lambda} #times (#hat{z} #times #hat{p}_{#Lambda})] , #vec{p}^{*}_{p} #upoint (#hat{z} #times #hat{p}_{#Lambda})); <#it{R}>/#hat{t}_{z}", kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("FakePolDrivers/ProxyZ/LeadJet/p2dRingOverProxyZ_VsProxyEtaVsCosThetaHEE", "<#it{R}>/#hat{t}_{z}; #eta_{Jet}; cos(#theta)=#hat{p}^{*}_{D} . #vec{p}_{#Lambda}; <#it{R}>/#hat{t}_{z}", kTProfile2D, {axisConfigurations.axisEtaCoarse, axisConfigurations.axisCosTheta});
      histos.add("FakePolDrivers/ProxyZ/LeadJet/p2dRingOverProxyZ_VsProxyEtaVsPhiStar", "<#it{R}>/#hat{t}_{z}; #eta_{Jet}; #phi^{*}; <#it{R}>/#hat{t}_{z}", kTProfile2D, {axisConfigurations.axisEtaCoarse, axisConfigurations.axisDeltaPhi});
    } // end doFakePolDiagnosticsQA bookings

    // Integrated observable for events with NLambda+NAntiLambda V0s per event
    // (an interesting measurement of correlation between <R> and Lambda-like V0s multiplicity. A proxy of covariance)
    // (calculated for leading jets only)
    histos.add("IntegratedCuts/LeadJet/pRingVsNV0s", "pRingVsNV0s; N_{#Lambda}+N_{#bar{#Lambda}};<#it{R}>", kTProfile, {{20, 0, 20}});                                                                  // See h2dPairedNV0sVsCentrality below for the correlation between number of V0s and centrality
    histos.add("IntegratedCuts/LeadJet/h2dPairedNV0sVsCentrality", "h2dPairedNV0sVsCentrality; N_{#Lambda}+N_{#bar{#Lambda}};Centrality (%)", kTH2D, {{20, 0, 20}, axisConfigurations.axisCentrality}); // One entry per collision with a valid leading jet

    // R_z-only diagnostics -- Compares the 3D ring, the projected ring and other useful diagnostics:
    if (useRingZ || useRingZT) {
      // The longitudinal ring in the titles: R_z, or R_z^T for useRingZT (the helicity-term titles name both on purpose):
      const std::string ringZSymbol = transverseBeamAxisMode ? "#it{R}_{z}^{T}" : "#it{R}_{z}";
      histos.add("RzDiagnostics/LeadJet/pRingVsChi", ("<" + ringZSymbol + "> vs #chi (LeadJet) (#hat{n} = cos(#chi)#hat{#phi} + sin(#chi)#hat{#theta});#chi_{LeadJet};<" + ringZSymbol + ">").c_str(), kTProfile, {axisConfigurations.axisDeltaPhi}); // chi is the angle such that n_hat = cos(chi) phi_hat + sin(chi) theta_hat.
      histos.add("RzDiagnostics/LeadP/pRingVsChi", ("<" + ringZSymbol + "> vs #chi (LeadP) (#hat{n} = cos(#chi)#hat{#phi} + sin(#chi)#hat{#theta});#chi_{LeadP};<" + ringZSymbol + ">").c_str(), kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("RzDiagnostics/SubJet/pRingVsChi", ("<" + ringZSymbol + "> vs #chi (SubJet) (#hat{n} = cos(#chi)#hat{#phi} + sin(#chi)#hat{#theta});#chi_{SubJet};<" + ringZSymbol + ">").c_str(), kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("RzDiagnostics/LeadJet/p2dRingVsChiVsMass", ("<" + ringZSymbol + "> vs #chi vs Mass (LeadJet) (#hat{n} = cos(#chi)#hat{#phi} + sin(#chi)#hat{#theta});#chi_{LeadJet};m_{p#pi} (GeV/c^{2});<" + ringZSymbol + ">").c_str(), kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("RzDiagnostics/LeadP/p2dRingVsChiVsMass", ("<" + ringZSymbol + "> vs #chi vs Mass (LeadP) (#hat{n} = cos(#chi)#hat{#phi} + sin(#chi)#hat{#theta});#chi_{LeadP};m_{p#pi} (GeV/c^{2});<" + ringZSymbol + ">").c_str(), kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("RzDiagnostics/SubJet/p2dRingVsChiVsMass", ("<" + ringZSymbol + "> vs #chi vs Mass (SubJet) (#hat{n} = cos(#chi)#hat{#phi} + sin(#chi)#hat{#theta});#chi_{SubJet};m_{p#pi} (GeV/c^{2});<" + ringZSymbol + ">").c_str(), kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("RzDiagnostics/LeadJet/p2dRingVsDeltaPhiVsDeltaEta", ("<" + ringZSymbol + "> vs (#Delta#varphi_{jet}, #Delta#eta_{jet});#Delta#varphi_{jet}=#phi_{#Lambda} - #phi_{Jet};#eta_{#Lambda}-#eta_{Jet};<" + ringZSymbol + ">").c_str(), kTProfile2D, {axisConfigurations.axisDeltaPhiCoarse, axisConfigurations.axisDeltaEtaCoarse});
      // R_perp = R - R_z, filled directly to store the proper error propagation (R and R_z share every candidate, thus are correlated):
      histos.add("RzDiagnostics/pRingPerpIntegrated", ("<#it{R}_{#perp}> per proxy; ;<#it{R}_{#perp}> = <#it{R}> - <" + ringZSymbol + ">").c_str(), kTProfile, {{3, 0, 3}});
      histos.get<TProfile>(HIST("RzDiagnostics/pRingPerpIntegrated"))->GetXaxis()->SetBinLabel(1, "LeadJet");
      histos.get<TProfile>(HIST("RzDiagnostics/pRingPerpIntegrated"))->GetXaxis()->SetBinLabel(2, "LeadP");
      histos.get<TProfile>(HIST("RzDiagnostics/pRingPerpIntegrated"))->GetXaxis()->SetBinLabel(3, "SubJet");
      histos.add("RzDiagnostics/LeadJet/pRingPerpVsMass", ("<#it{R}_{#perp}> vs Mass (LeadJet);m_{p#pi} (GeV/c^{2});<#it{R}_{#perp}> = <#it{R}> - <" + ringZSymbol + ">").c_str(), kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
      histos.add("RzDiagnostics/LeadP/pRingPerpVsMass", ("<#it{R}_{#perp}> vs Mass (LeadP);m_{p#pi} (GeV/c^{2});<#it{R}_{#perp}> = <#it{R}> - <" + ringZSymbol + ">").c_str(), kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
      histos.add("RzDiagnostics/SubJet/pRingPerpVsMass", ("<#it{R}_{#perp}> vs Mass (SubJet);m_{p#pi} (GeV/c^{2});<#it{R}_{#perp}> = <#it{R}> - <" + ringZSymbol + ">").c_str(), kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
      histos.add("RzDiagnostics/LeadJet/pRingPerpVsPhiAEE", ("<#it{R}_{#perp}> vs AEE angle (LeadJet);#phi_{#Lambda-like}-#phi_{p-like}^{*};<#it{R}_{#perp}> = <#it{R}> - <" + ringZSymbol + ">").c_str(), kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("RzDiagnostics/LeadP/pRingPerpVsPhiAEE", ("<#it{R}_{#perp}> vs AEE angle (LeadP);#phi_{#Lambda-like}-#phi_{p-like}^{*};<#it{R}_{#perp}> = <#it{R}> - <" + ringZSymbol + ">").c_str(), kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("RzDiagnostics/SubJet/pRingPerpVsPhiAEE", ("<#it{R}_{#perp}> vs AEE angle (SubJet);#phi_{#Lambda-like}-#phi_{p-like}^{*};<#it{R}_{#perp}> = <#it{R}> - <" + ringZSymbol + ">").c_str(), kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("RzDiagnostics/LeadJet/pRingPerpVsDeltaPhi", ("<#it{R}_{#perp}> vs #Delta#varphi (LeadJet);#Delta#varphi_{jet};<#it{R}_{#perp}> = <#it{R}> - <" + ringZSymbol + ">").c_str(), kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("RzDiagnostics/LeadP/pRingPerpVsDeltaPhi", ("<#it{R}_{#perp}> vs #Delta#varphi (LeadP);#Delta#varphi_{LeadP};<#it{R}_{#perp}> = <#it{R}> - <" + ringZSymbol + ">").c_str(), kTProfile, {axisConfigurations.axisDeltaPhi});
      histos.add("RzDiagnostics/SubJet/pRingPerpVsDeltaPhi", ("<#it{R}_{#perp}> vs #Delta#varphi (SubJet);#Delta#varphi_{SubJet};<#it{R}_{#perp}> = <#it{R}> - <" + ringZSymbol + ">").c_str(), kTProfile, {axisConfigurations.axisDeltaPhi});
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
      histos.add("RzDiagnostics/LeadJet/p2dRingVsDeltaPhiVsMassEtaPos", ("<" + ringZSymbol + "> vs (#Delta#varphi_{jet}, mass), #eta_{#Lambda} > 0;#Delta#varphi_{jet};m_{p#pi} (GeV/c^{2});<" + ringZSymbol + ">").c_str(), kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("RzDiagnostics/LeadJet/p2dRingVsDeltaPhiVsMassEtaNeg", ("<" + ringZSymbol + "> vs (#Delta#varphi_{jet}, mass), #eta_{#Lambda} < 0;#Delta#varphi_{jet};m_{p#pi} (GeV/c^{2});<" + ringZSymbol + ">").c_str(), kTProfile2D, {axisConfigurations.axisDeltaPhi, axisConfigurations.axisLambdaMassSigExtract});
      histos.add("RzDiagnostics/LeadJet/pCosSqThetaStarZVsMass", "<cos^{2}#theta^{*}_{z}> vs Mass (LeadJet);m_{p#pi} (GeV/c^{2});<cos^{2}#theta^{*}_{z}>", kTProfile, {axisConfigurations.axisLambdaMassSigExtract});
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
    if (doMixingQA) {
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

      histos.add("EventMixingQA/AngularCorrelations/LeadP/h2dMixedProxyEtaVsProxyEta", "MixedLeadP #eta vs LeadP #eta;MixedLeadP #eta; LeadP #eta;Counts", kTH2D, {axisConfigurations.axisEtaCoarse, axisConfigurations.axisEtaCoarse});
      histos.add("EventMixingQA/AngularCorrelations/LeadJet/h2dMixedProxyEtaVsProxyEta", "MixedLeadJet #eta vs LeadJet #eta;MixedLeadJet #eta; LeadJet #eta;Counts", kTH2D, {axisConfigurations.axisEtaCoarse, axisConfigurations.axisEtaCoarse});
      histos.add("EventMixingQA/AngularCorrelations/SubJet/h2dMixedProxyEtaVsProxyEta", "MixedSubJet #eta vs SubJet #eta;MixedSubJet #eta; SubJet #eta;Counts", kTH2D, {axisConfigurations.axisEtaCoarse, axisConfigurations.axisEtaCoarse});
      histos.add("EventMixingQA/AngularCorrelations/LeadP/h2dMixedProxyPhiVsProxyPhi", "MixedLeadP #phi vs LeadP #phi;MixedLeadP #phi; LeadP #phi;Counts", kTH2D, {axisConfigurations.axisPhi, axisConfigurations.axisPhi});
      histos.add("EventMixingQA/AngularCorrelations/LeadJet/h2dMixedProxyPhiVsProxyPhi", "MixedLeadJet #phi vs LeadJet #phi;MixedLeadJet #phi; LeadJet #phi;Counts", kTH2D, {axisConfigurations.axisPhi, axisConfigurations.axisPhi});
      histos.add("EventMixingQA/AngularCorrelations/SubJet/h2dMixedProxyPhiVsProxyPhi", "MixedSubJet #phi vs SubJet #phi;MixedSubJet #phi; SubJet #phi;Counts", kTH2D, {axisConfigurations.axisPhi, axisConfigurations.axisPhi});

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
      if (doResamplingPassQA) {
        const int nPasses = fakePolSwitches.nProxyResamples;
        histos.add("EventMixingQA/CollLoopOutcome/LeadP/hMixedEventDistinctSources", "hMixedEventDistinctSources;Distinct sources stored;Counts", kTH1D, {{nPasses + 1, -0.5, nPasses + 0.5}});
        histos.add("EventMixingQA/CollLoopOutcome/LeadJet/hMixedEventDistinctSources", "hMixedEventDistinctSources;Distinct sources stored;Counts", kTH1D, {{nPasses + 1, -0.5, nPasses + 0.5}});
        histos.add("EventMixingQA/CollLoopOutcome/SubJet/hMixedEventDistinctSources", "hMixedEventDistinctSources;Distinct sources stored;Counts", kTH1D, {{nPasses + 1, -0.5, nPasses + 0.5}});
      }
    }

    // Fetch the X-axes of the Delta Method histograms booked above:
    mAxisPt = histosRing.get<TH2>(HIST("Ring/DeltaMethod/h2dLambdaPtVsDeltaComp"))->GetXaxis();
    mAxisMass = histosRing.get<TH2>(HIST("Ring/DeltaMethod/h2dMassVsDeltaComp"))->GetXaxis();
    mAxisDTheta = histosRing.get<TH2>(HIST("Ring/DeltaMethod/h2dDeltaThetaVsDeltaComp"))->GetXaxis();
    deltaTracker.resize(mAxisPt->GetNbins(), mAxisMass->GetNbins(), mAxisDTheta->GetNbins());
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
      histos->fill(HIST("SelectionFlow/hAnalysisLevelSelectionV0s"), ++binValue); // Hardcoded names, as they will not change. Increments before filling, by default
      histos->fill(HIST("SelectionFlow/h2dAnalysisLevelSelectionV0sVsMass"), binValue, massV0);
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
    histos.fill(HIST("SelectionFlow/h2dArmenterosInput"), v0.alpha(), v0.qtArm());
 
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
    histos.fill(HIST("SelectionFlow/h2dArmenterosAnalysisCuts"), v0.alpha(), v0.qtArm());
    return true;
  }

  /// \brief A function that applies analysisLevelCutsJets' acceptance to one jet proxy (leading jet, subleading jet or leading particle).
  /// \note  Called together with the minimum-pT selections, before any V0 processing. Always true when doAnalysisLevelCutsJets is off.
  bool isProxyAccepted(float proxyEta, float maxAbsEta)
  {
    if (!analysisLevelCutsJets.doAnalysisLevelCutsJets)
      return true;
    if (!(std::abs(proxyEta) < maxAbsEta)) // Written as !(value < threshold) so a NaN eta is rejected
      return false;

    // Quenching-related cuts
    // (TODO)

    return true;
  }

  // Initializing a random number generator for the worker (for perpendicular-to-jet direction QAs):
  TRandom3 randomGen{0}; // 0 means we auto-seed from machine entropy. This is called once per device in the pipeline, so we should not see repeated seeds across workers
  std::mt19937 rng{std::random_device{}()};

  // Pre-computed values for helper below:
  double smearSigma = 0.; // Set in init: a member initializer runs before the JSON is applied, so it would always see the default jetR
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
  /// \param v0RingInputs the per-V0 quantities, computed once and shared by every proxy.
  /// \note  R = (3/alpha) p_D_hat^* \cdot n, with n = (t_hat x p_Lambda)/|t_hat x p_Lambda|. The longitudinal modes keep only
  ///        the beam-axis part of p_D^* (R_z) or that part made transverse to the Lambda (R_z^T).
  /// \note  Call it only for a valid proxy: a missing proxy keeps the ProxyRingCache defaults.
  ProxyRingCache computeProxyRing(const XYZVector& proxyUnitVec, float proxyPhi, const V0RingInputs& v0RingInputs)
  {
    ProxyRingCache ringCache;
    XYZVector cross = proxyUnitVec.Cross(v0RingInputs.lambdaLike3Vec);
    const float invCrossNorm = 1.f / cross.R(); // Caching the .R() result
    const float ringFullUnit = v0RingInputs.protonLikeStarUnit3Vec.Dot(cross) * invCrossNorm; // v: the unitless full ring
    if (!longitudinalRingMode) {
      ringCache.ringObservable = ringFullUnit;
    } else {
      ringCache.nz = cross.Z() * invCrossNorm;
      ringCache.ringObservable = v0RingInputs.protonStarRingBeamComponent * ringCache.nz;
      ringCache.kappaDen = ringCache.nz * ringCache.nz;
      // Diagnostics only: the complement R - R_long, the helicity term R_z - R_z^T, and the bearing chi:
      ringCache.ringPerp = v0RingInputs.ringPrefactor * (ringFullUnit - ringCache.ringObservable);
      ringCache.helicityTerm = v0RingInputs.ringPrefactor * v0RingInputs.protonStarHelicityBeamComponent * ringCache.nz;
      ringCache.chi = std::atan2(-cross.Z() * v0RingInputs.v0p, cross.Y() * v0RingInputs.v0px - cross.X() * v0RingInputs.v0py);
    }
    ringCache.kappaNum = ringCache.ringObservable * ringFullUnit; // u v (u^2 for the full ring)
    // Adding the prefactor related to the CP-violating decay (decay constants have different signs). Prefactor goes last!
    ringCache.ringObservable *= v0RingInputs.ringPrefactor;
    // Angular variables
    ringCache.deltaPhi = wrapToPiFast(v0RingInputs.v0phi - proxyPhi);            // Wrapped to [-PI, PI), for convenience
    ringCache.cosDeltaTheta = proxyUnitVec.Dot(v0RingInputs.lambdaLikeUnit3Vec); // Uses the pre-calculated unit vectors to avoid recomputation
    ringCache.deltaTheta = std::acos(ringCache.cosDeltaTheta);                   // 3D angular separation. Same as ROOT::Math::VectorUtil::Angle(proxyUnitVec, lambdaLike3Vec);
    return ringCache;
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

  /// \brief doMixedEventProxies LUT of one proxy type: pools the collisions by mixing bin, then pairs each collision with the next
  ///        mixedEventWindowSize partners inside its own bin, never itself, feeding the reservoirs of both source/target at once.
  /// \note  Reproduces CombinationsBlockStrictlyUpperSameIndexPolicy exactly, in the same bins, same forward window,
  ///        strictly upper. The advantage is that it runs in O(M*W) rather than the O(M^2) time (M being the collisions carrying a valid proxy, W the window size).
  /// \return The number of pairs formed, for QA.
  template <int idxProxy>
  int64_t buildMixingLut(ProxyByCollision& proxyByCollision, BinningPolicyBase<3>& mixBinning, const std::vector<MixingAxes>& mixingAxesByCollision, int64_t collisionIndexBase)
  {
    int64_t nPairs = 0;
    const auto pools = poolByBin(mixBinning, proxyByCollision.proxyKinematicsVec, mixingAxesByCollision);
    for (auto const& pool : pools) {
      for (size_t a = 0; a < pool.size(); ++a) {
        const size_t slot1 = pool[a];
        const int64_t id1 = collisionIndexBase + static_cast<int64_t>(slot1);
        const size_t lastPartner = std::min(a + static_cast<size_t>(fakePolSwitches.mixedEventWindowSize) + 1, pool.size());
        // Forward window occupancy, i.e. how many partners this collision is about to be paired with
        // (should reproduce currentWindowNeighbours(), saturating at mixedEventWindowSize):
        if (doMixingQA)
          histos.fill(HIST("EventMixingQA/") + HIST(ProxyNames[idxProxy]) + HIST("/hMixedEventWindowNeighbours"), lastPartner - a - 1);
        for (size_t b = a + 1; b < lastPartner; ++b) {
          const size_t slot2 = pool[b];
          const int64_t id2 = collisionIndexBase + static_cast<int64_t>(slot2);
          // Continuous-readout cannibalisation test:
          // (should not be impactful with gap ~ 10, as close-index collisions are most likely not close in time on the derived data, but kept as QA)
          if (fakePolSwitches.mixingIdxGapSize > 0 && std::abs(id2 - id1) <= fakePolSwitches.mixingIdxGapSize)
            continue;
          // Both sides of the pair feed each other's reservoir, properly using what is already in memory. This also lets a collision borrow from
          // earlier collisions as well, eliminating the forward-only bias of the sliding window in the event mixing procedure.
          reservoirInsert(proxyByCollision, slot1, id1, slot2, id2);
          reservoirInsert(proxyByCollision, slot2, id2, slot1, id1);
          if (doMixingQA)
            histos.fill(HIST("EventMixingQA/IndexQA/") + HIST(ProxyNames[idxProxy]) + HIST("/hMixedEventDeltaIndexEligible"), id2 - id1);
          ++nPairs;
        }
      }
    }
    return nPairs;
  }

  /// \brief Source-usage QA of one proxy's finished LUT, over every stored entry (one per borrowing/resampling pass).
  /// \note Each target owns nResamples consecutive entries (filled from the first, -1 when empty), so "entry / nResamples" is always the target slot.
  template <int idxProxy>
  void fillMixingSourceQA(const ProxyByCollision& proxyByCollision)
  {
    const size_t nResamples = static_cast<size_t>(fakePolSwitches.nProxyResamples);
    for (size_t entry = 0; entry < proxyByCollision.mixedSourcesVec.size(); ++entry) {
      const int32_t sourceSlot = proxyByCollision.mixedSourcesVec[entry];
      if (sourceSlot < 0) continue;
      histos.fill(HIST("EventMixingQA/IndexQA/") + HIST(ProxyNames[idxProxy]) + HIST("/hMixedEventDeltaIndexSelected"), static_cast<int64_t>(entry / nResamples) - sourceSlot);
    }

    // One fill per source, at its own eta/phi (the slot reaches its cache directly, so no LUT re-scan is needed):
    constexpr auto folderPath = HIST("EventMixingQA/SourceUsage/") + HIST(ProxyNames[idxProxy]);
    for (auto const& kv : tallySourceUsage(proxyByCollision.mixedSourcesVec)) {
      auto const& source = proxyByCollision.proxyKinematicsVec[kv.first];
      histos.fill(folderPath + HIST("/hMixedEventSourceUsageCount"), kv.second);
      histos.fill(folderPath + HIST("/pMixedEventSourceUsageVsEta"), source.eta, kv.second);
      histos.fill(folderPath + HIST("/pMixedEventSourceUsageVsPhi"), source.phi, kv.second);
    }
  }

  /// \brief The five Delta Method components (r_k, n_k, r_k^2, n_k^2, r_k*n_k) of every non-empty bin of one differential accumulator.
  template <typename THistName>
  void fillDeltaMethodComponents(const THistName& histName, const std::vector<double>& rVec, const std::vector<int>& nVec, TAxis* axis)
  {
    for (size_t bin = 0; bin < rVec.size(); ++bin) {
      int nVal = nVec[bin];
      if (nVal == 0)
        continue;
      double rVal = rVec[bin];
      double center = axis->GetBinCenter(bin);
      histosRing.fill(histName, center, 0.5, rVal);
      histosRing.fill(histName, center, 1.5, static_cast<double>(nVal));
      histosRing.fill(histName, center, 2.5, rVal * rVal);
      histosRing.fill(histName, center, 3.5, static_cast<double>(nVal * nVal));
      histosRing.fill(histName, center, 4.5, rVal * nVal);
    }
  }

  // Defining filters for events:
  Filter zvtxFilter = (nabs(o2::aod::lambdajetpol::zvtx) < maxZVtxPosition);

  // Preslices for correct collisions association:
  // (tested custom grouping and performs worse here)
  Preslice<aod::RingJets> perColJets = o2::aod::lambdajetpol::ringCollisionId;
  Preslice<aod::RingLaV0s> perColV0s = o2::aod::lambdajetpol::ringCollisionId;
  Preslice<aod::RingLeadPs> perColLeadPs = o2::aod::lambdajetpol::ringCollisionId;
  /// \brief Main analysis loop: for each collision, rebuilds the leading jet/particle proxies (with optional fakePolSwitches distortions), then loops over V0s computing the ring observable and polarization-vector profiles.
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
      leadJet.isValid = leadJet.pt > analysisLevelCutsJets.minLeadJetPt;
      subJet.isValid = subJet.pt > analysisLevelCutsJets.minSubJetPt;
      if (leadJet.isValid) {
        leadJet.eta = leadJetRow->jetEta();
        leadJet.phi = leadJetRow->jetPhi();
        leadJet.isValid = isProxyAccepted(leadJet.eta, analysisLevelCutsJets.maxLeadJetEta); // Jet acceptance (analysisLevelCutsJets)
      }
      if (subJet.isValid) {
        subJet.eta = subJetRow->jetEta();
        subJet.phi = subJetRow->jetPhi();
        subJet.isValid = isProxyAccepted(subJet.eta, analysisLevelCutsJets.maxSubJetEta);
      }
      leadJetByCollision.proxyKinematicsVec[slot] = leadJet;
      subJetByCollision.proxyKinematicsVec[slot] = subJet;

      // The leading particle is resolved here too, so the pooling and the main loop read a vector instead of slicing RingLeadPs.
      // A collision with no leadP simply keeps isValid = false, and its mixing axes are stored regardless
      for (auto const& lp : leadPs.sliceBy(perColLeadPs, collId)) {
        const bool isValidLeadP = lp.leadParticlePt() > analysisLevelCutsJets.minLeadPPt && isProxyAccepted(lp.leadParticleEta(), analysisLevelCutsJets.maxLeadPEta); // Minimum pT (not necessarily the same as in derived data builder. Can be a stricter cut!) and acceptance
        leadPByCollision.proxyKinematicsVec[slot] = {isValidLeadP, lp.leadParticlePt(), lp.leadParticleEta(), lp.leadParticlePhi()};
        break; // Table has at most one LeadP per collision, but we break nonetheless
      }
    }

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

      // Starting the window loops for each jet proxy:
      // Leading particle:
      const int64_t nPairsLeadP = buildMixingLut<kLeadP>(leadPByCollision, mixBinning, mixingAxesByCollision, collisionIndexBase);
      const auto tLeadPDone = std::chrono::steady_clock::now();
      LOG(info) << "  leadP loop: " << std::chrono::duration<double>(tLeadPDone - tLutStart).count() << " s, " << nPairsLeadP << " pairs";

      // Leading jet:
      const int64_t nPairsLeadJet = buildMixingLut<kLeadJet>(leadJetByCollision, mixBinning, mixingAxesByCollision, collisionIndexBase);
      const auto tLeadJetDone = std::chrono::steady_clock::now();
      LOG(info) << "  leadJet loop: " << std::chrono::duration<double>(tLeadJetDone - tLeadPDone).count() << " s, " << nPairsLeadJet << " pairs";

      // Subleading jet:
      const int64_t nPairsSubJet = buildMixingLut<kSubJet>(subJetByCollision, mixBinning, mixingAxesByCollision, collisionIndexBase);
      const auto tSubJetDone = std::chrono::steady_clock::now();
      LOG(info) << "  subJet loop: " << std::chrono::duration<double>(tSubJetDone - tLeadJetDone).count() << " s, " << nPairsSubJet << " pairs";

      // Source-usage QA, over every stored entry:
      if (doMixingQA) {
        fillMixingSourceQA<kLeadP>(leadPByCollision);
        fillMixingSourceQA<kLeadJet>(leadJetByCollision);
        fillMixingSourceQA<kSubJet>(subJetByCollision);
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

        // Slice the V0s belonging to this collision (jets and leading particle were resolved once, in the pre-pass):
        // (global collision indices repeat a lot, but they are unique to a same TimeFrame (TF) subfolder in the derived data)
        auto v0sInColl = v0s.sliceBy(perColV0s, collId);

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

        ////////////////////
        // Per-proxy helpers of this collision (lambdas, so they read the collision variables above directly):
        // Each takes a ProxyTypeTag, so the proxy type is known at compile time.

        // doMixedEventProxies helper: returns the proxy this collision borrows in this resampling pass, rebuilt from the LUT above.
        // An empty entry gets "isValid = false" if there is no valid mixing target, or fewer distinct sources than resampling passes.
        // (a target with k < nProxyResamples distinct sources is still used in its first k passes: dropping it would cost the sparse, high proxy pT bins)
        auto borrowMixedProxy = [&](auto proxyTypeTag, const ProxyKinematics& proxy, const ProxyByCollision& proxyByCollision) {
          // Gets the enum value at compile time (here only for the proxy name in the LOG below):
          constexpr int idxProxy = decltype(proxyTypeTag)::value;
          const MixedProxyInfo mixedProxyInfo = rebuildBorrowedProxyFromCache(proxyByCollision, mixingAxesByCollision, collisionIndexBase, collSlot, idxResampling); // Fetching information for this collId's event mixing source
          if (mixedProxyInfo.proxy.isValid) {
            // bit-by-bit checks of correspondence (asked for no-lint on these "paranoid" checks):
            // (inexpensive enough that we can check this at production time)
            const bool samePtProxy = (mixedProxyInfo.proxy.pt == proxy.pt);    // NOLINT(clang-diagnostic-float-equal)
            const bool sameEtaProxy = (mixedProxyInfo.proxy.eta == proxy.eta); // NOLINT(clang-diagnostic-float-equal)
            const bool samePhiProxy = (mixedProxyInfo.proxy.phi == proxy.phi); // NOLINT(clang-diagnostic-float-equal)
            if (samePtProxy && sameEtaProxy && samePhiProxy) {                 // Astronomically small chance, even with the birthday paradox
              LOG(fatal) << "EventMixing: borrowed " << ProxyNames[idxProxy] << " is bit-identical to the target's own in pt, eta and phi. Target collision "
                         << collId << ", source " << mixedProxyInfo.sourceCollisionId << ", pt " << proxy.pt;
            }
          }
          return mixedProxyInfo;
        };

        // doMixedEventProxies QA of one borrowed proxy against this collision's own proxy (before any distortion):
        auto fillMixingOutcomeQA = [&](auto proxyTypeTag, const ProxyKinematics& proxy, const MixedProxyInfo& mixedProxyInfo, const ProxyByCollision& proxyByCollision) {
          // Gets the enum value at compile time via decltype's ::value, so HIST() can use it (a for loop over the enum would only give it at runtime):
          constexpr int idxProxy = decltype(proxyTypeTag)::value;
          constexpr auto outcomeFolderPath = HIST("EventMixingQA/CollLoopOutcome/") + HIST(ProxyNames[idxProxy]);
          constexpr auto deltasFolderPath = HIST("EventMixingQA/SourceTargetDeltas/") + HIST(ProxyNames[idxProxy]);
          // Filled hit or miss, so the zero bin is the miss rate seen from the pool's side (once per collision, not per resampling pass):
          if (idxResampling == 0) {
            const int nCandidates = proxyByCollision.candidateCountVec[collSlot]; // Zero-initialised, so no existence check needed
            histos.fill(outcomeFolderPath + HIST("/hMixedEventCandidates"), nCandidates);
            histos.fill(outcomeFolderPath + HIST("/h2dMixedCandidatesVsPt"), nCandidates, proxy.pt);
            if (doResamplingPassQA)
              histos.fill(outcomeFolderPath + HIST("/hMixedEventDistinctSources"), distinctSourcesOf(proxyByCollision, collSlot));
          }

          if (!mixedProxyInfo.proxy.isValid) {
            histos.fill(outcomeFolderPath + HIST("/hMixedEventOutcome"), 0);
            histos.fill(outcomeFolderPath + HIST("/h2dMixedEventOutcomeVsTargetZVtx"), 0, collisionPVz);
            histos.fill(outcomeFolderPath + HIST("/h2dMixedEventOutcomeVsTargetCentrality"), 0, centrality);
            histos.fill(outcomeFolderPath + HIST("/h2dMixedEventOutcomeVsTargetProxyPt"), 0, proxy.pt);
            return;
          }
          // Only fill the comparison histograms on an actual hit (source vs. this collision's own):
          histos.fill(outcomeFolderPath + HIST("/hMixedEventOutcome"), 1);
          histos.fill(outcomeFolderPath + HIST("/h2dMixedEventOutcomeVsTargetZVtx"), 1, collisionPVz);
          histos.fill(outcomeFolderPath + HIST("/h2dMixedEventOutcomeVsTargetCentrality"), 1, centrality);
          histos.fill(outcomeFolderPath + HIST("/h2dMixedEventOutcomeVsTargetProxyPt"), 1, proxy.pt);

          histos.fill(HIST("EventMixingQA/AngularCorrelations/") + HIST(ProxyNames[idxProxy]) + HIST("/h2dMixedProxyEtaVsProxyEta"), mixedProxyInfo.proxy.eta, proxy.eta);
          histos.fill(HIST("EventMixingQA/AngularCorrelations/") + HIST(ProxyNames[idxProxy]) + HIST("/h2dMixedProxyPhiVsProxyPhi"), mixedProxyInfo.proxy.phi, proxy.phi);
          histos.fill(deltasFolderPath + HIST("/hMixedEventDeltaPt"), mixedProxyInfo.proxy.pt - proxy.pt);
          histos.fill(deltasFolderPath + HIST("/hMixedEventDeltaZvtx"), mixedProxyInfo.zvtx - collisionPVz);
          histos.fill(deltasFolderPath + HIST("/hMixedEventDeltaCent"), mixedProxyInfo.centrality - centrality);

          histos.fill(deltasFolderPath + HIST("/hMixedEventDeltaEta"), mixedProxyInfo.proxy.eta - proxy.eta);
          histos.fill(deltasFolderPath + HIST("/hMixedEventDeltaPhi"), wrapToPiFast(mixedProxyInfo.proxy.phi - proxy.phi));

          if constexpr (idxProxy == kLeadP) { // (LeadP only, as the other proxies would verify the same thing)
            // QAing the binning policy itself to verify if any wrong proxies are being borrowed:
            histos.fill(deltasFolderPath + HIST("/h2dMixedEventDeltaPtVsTargetPt"), mixedProxyInfo.proxy.pt - proxy.pt, proxy.pt);
            histos.fill(deltasFolderPath + HIST("/h2dMixedEventDeltaZvtxVsTargetZvtx"), mixedProxyInfo.zvtx - collisionPVz, collisionPVz);
            histos.fill(deltasFolderPath + HIST("/h2dMixedEventDeltaCentVsTargetCent"), mixedProxyInfo.centrality - centrality, centrality);

            // 2D Index Vs Delta Mixed-Variable QAs:
            const float deltaCollisionIndex = static_cast<float>(collId - mixedProxyInfo.sourceCollisionId);
            histos.fill(HIST("EventMixingQA/IndexQA/LeadP/h2dMixedEventDeltaPtVsDeltaIndex"), deltaCollisionIndex, mixedProxyInfo.proxy.pt - proxy.pt);
            histos.fill(HIST("EventMixingQA/IndexQA/LeadP/h2dMixedEventDeltaZvtxVsDeltaIndex"), deltaCollisionIndex, mixedProxyInfo.zvtx - collisionPVz);
            histos.fill(HIST("EventMixingQA/IndexQA/LeadP/h2dMixedEventDeltaCentVsDeltaIndex"), deltaCollisionIndex, mixedProxyInfo.centrality - centrality);

            // Bit-by-bit coincidences (the same checks as the guard in borrowMixedProxy):
            const bool samePtProxy = (mixedProxyInfo.proxy.pt == proxy.pt);    // NOLINT(clang-diagnostic-float-equal)
            const bool sameEtaProxy = (mixedProxyInfo.proxy.eta == proxy.eta); // NOLINT(clang-diagnostic-float-equal)
            const bool samePhiProxy = (mixedProxyInfo.proxy.phi == proxy.phi); // NOLINT(clang-diagnostic-float-equal)
            const bool sameZvtx = (mixedProxyInfo.zvtx == collisionPVz);       // NOLINT(clang-diagnostic-float-equal)

            if (sameZvtx && samePtProxy) // Flagged as "alarm" level
              LOG(alarm) << "EventMixing: source and target share Zvtx and leadP pt bit-for-bit -- duplicated collision row? Target "
                         << collId << ", source " << mixedProxyInfo.sourceCollisionId << ", Zvtx " << collisionPVz;

            // Single-variable coincidences are possible by chance, so they are counted instead:
            if (samePtProxy)
              histos.fill(HIST("EventMixingQA/IdentityChecks/LeadP/hMixedEventIdentityFlags"), 0);
            if (sameEtaProxy)
              histos.fill(HIST("EventMixingQA/IdentityChecks/LeadP/hMixedEventIdentityFlags"), 1);
            if (samePhiProxy)
              histos.fill(HIST("EventMixingQA/IdentityChecks/LeadP/hMixedEventIdentityFlags"), 2);
            if (sameZvtx)
              histos.fill(HIST("EventMixingQA/IdentityChecks/LeadP/hMixedEventIdentityFlags"), 3);

            // Possible shared-track case in event splitting, done in a log scale:
            const float angularSeparation = std::hypot(mixedProxyInfo.proxy.eta - proxy.eta, wrapToPiFast(mixedProxyInfo.proxy.phi - proxy.phi));
            histos.fill(HIST("EventMixingQA/IdentityChecks/LeadP/hMixedEventLogAngularSep"), angularSeparation > 0.f ? std::log10(angularSeparation) : -7.f);
          }
        };

        // KinematicsQA/<Proxy>/ of the proxy actually used (after applyProxyDistortion):
        auto fillProxyKinematicsQA = [&](auto proxyTypeTag, const ProxyKinematics& proxy) {
          // Gets the enum value at compile time via decltype's ::value, so HIST() can use it (a for loop over the enum would only give it at runtime):
          constexpr int idxProxy = decltype(proxyTypeTag)::value;
          constexpr auto folderPath = HIST("KinematicsQA/") + HIST(ProxyNames[idxProxy]);
          histos.fill(folderPath + HIST("/hEta"), proxy.eta);
          histos.fill(folderPath + HIST("/hPhi"), proxy.phi);
          histos.fill(folderPath + HIST("/hPt"), proxy.pt);

          histos.fill(folderPath + HIST("/h2dEtaVsPhi"), proxy.eta, proxy.phi);
          histos.fill(folderPath + HIST("/h2dEtaVsPVz"), proxy.eta, collisionPVz);
        };
        // end of per-proxy fill helpers of this collision
        ////////////////////

        // 2) We require at least one leading particle:
        // (The goal is to see how diluted the signal gets with events which don't even have a loose FastJet jet)
        // (The leading particle is built from all tracks that passed the pseudojet
        // selection, so it exists whenever FastJet was run on this collision.
        // Events that have a leading jet always have a leading particle too, but
        // the converse is not true: events can have a leading particle with no jet
        // if no jet survives the pT threshold/the background subtraction)
        // (At least that is the case when minLeadPPt = 0)
        // Fetching the leading particle -- Resolved once per collision in the leadPByCollision pre-pass (minimum pT and acceptance included):
        // (Local copy, as applyProxyDistortion may change it for this collision only)
        ProxyKinematics leadP = leadPByCollision.proxyKinematicsVec[collSlot];

        // Snapshot this collision's own leading particle before any distortion can overwrite it,
        // so forcePreviousJet can hand it to the next collision:
        const ProxyKinematics beforeDistLeadP = leadP;

        // Build leading particle unit vector, outside the V0 loop for performance.
        XYZVector leadPUnitVec(1., 0., 0.); // dummy (overwritten below when leadP.isValid)
        float leadPZ = 0.;                  // \hat t_z of the leading-particle proxy actually used (see axisLeadPZ)
        if (leadP.isValid) {
          // Rebuild the direction from the cached eta/phi, as for the jets and the borrowed proxies (cheaper than going through px/py/pz and normalizing with .Unit()):
          const double inverseCoshEta = 1.0 / std::cosh(leadP.eta);
          leadPUnitVec = XYZVector(std::cos(leadP.phi) * inverseCoshEta, std::sin(leadP.phi) * inverseCoshEta, std::tanh(leadP.eta));

          // doMixedEventProxies: get this collision's mixing partner (if any) from the LUT built above:
          // (this check is performed only if leadP.isValid is true for performance, as the LeadPPt matching for the event mixing would also demand a minimum jet pT)
          // (it is also a physics selection: we want to mix the correlations in events that could actually have a ring formed)
          if (fakePolSwitches.doMixedEventProxies) {
            const MixedProxyInfo mixedLeadPInfo = borrowMixedProxy(ProxyTypeTag<kLeadP>, leadP, leadPByCollision);
            proxyCache.leadP = mixedLeadPInfo.proxy;
            if (doMixingQA)
              fillMixingOutcomeQA(ProxyTypeTag<kLeadP>, leadP, mixedLeadPInfo, leadPByCollision);
          }

          // Apply distortion logic:
          // (modifies (leadP, leadPUnitVec) as if the modified proxy was the actual proxy of this event)
          applyProxyDistortion(leadP, leadPUnitVec, analysisLevelCutsJets.minLeadPPt, fakePolSwitches.maxLeadPProxyEta, proxyCache.leadP, etaLeadPDist, phiLeadPDist);

          // Fill distorted-proxy QA histograms (i.e., the actually used proxy).
          // A borrowed-proxy miss and a failed pT gate will skip the fill.
          if (leadP.isValid) {
            fillProxyKinematicsQA(ProxyTypeTag<kLeadP>, leadP);
            leadPZ = leadPUnitVec.Z(); // After applyProxyDistortion, as for leadJetZ: the RingKernel needs the post-distortion direction the ring was computed with
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
            const MixedProxyInfo mixedLeadJetInfo = borrowMixedProxy(ProxyTypeTag<kLeadJet>, leadJet, leadJetByCollision);
            proxyCache.leadJet = mixedLeadJetInfo.proxy;
            if (doMixingQA)
              fillMixingOutcomeQA(ProxyTypeTag<kLeadJet>, leadJet, mixedLeadJetInfo, leadJetByCollision);
          }
          applyProxyDistortion(leadJet, leadJetUnitVec, analysisLevelCutsJets.minLeadJetPt, fakePolSwitches.maxJetProxyEta, proxyCache.leadJet, etaLeadPDist, phiLeadPDist);

          // Fill distorted-proxy QA histograms:
          if (leadJet.isValid) {
            fillProxyKinematicsQA(ProxyTypeTag<kLeadJet>, leadJet);
            histos.fill(HIST("IntegratedCuts/LeadJet/h2dPairedNV0sVsCentrality"), nLambdaLikeV0s, centrality); // Once per collision, like the KinematicsQA/ counters

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
            const MixedProxyInfo mixedSubJetInfo = borrowMixedProxy(ProxyTypeTag<kSubJet>, subJet, subJetByCollision);
            proxyCache.subJet = mixedSubJetInfo.proxy;
            if (doMixingQA)
              fillMixingOutcomeQA(ProxyTypeTag<kSubJet>, subJet, mixedSubJetInfo, subJetByCollision);
          }
          applyProxyDistortion(subJet, subJetUnitVec, analysisLevelCutsJets.minSubJetPt, fakePolSwitches.maxJetProxyEta, proxyCache.subJet, etaLeadPDist, phiLeadPDist);

          // Fill distorted-proxy QA histograms:
          if (subJet.isValid) {
            fillProxyKinematicsQA(ProxyTypeTag<kSubJet>, subJet);
          }
        }

        // forcePreviousJet: hand this collision's own proxies to the next collision.
        if (fakePolSwitches.forcePreviousJet) {
          prevJetCache = SetOfProxies{beforeDistLeadP, leadJetByCollision.proxyKinematicsVec[collSlot], subJetByCollision.proxyKinematicsVec[collSlot]};
        }

        // Hardcoded proxy |eta| cuts for bins 2 and 3 of IntegratedCuts/ (the old kinematic-cut families), independent from the analysisLevelCutsJets acceptance:
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
        deltaTracker.reset();
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
          // Naive estimator based on signal extractions from outside this code (the regions come from the NSigma configurables, as defined in massRegionOf()):
          const MassRegion v0MassRegion = massRegionOf(v0LambdaLikeMass);
          const bool v0InMassPeak = v0MassRegion == kPeak;
          const bool v0InMassWindow = v0MassRegion == kLeftSideband || v0MassRegion == kRightSideband;
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

              if (positiveGeom)
                histos.fill(HIST("FakePolDrivers/hMassDecayGeomRight"), v0LambdaLikeMass);
              else
                histos.fill(HIST("FakePolDrivers/hMassDecayGeomLeft"), v0LambdaLikeMass);
            }

            // Measuring the AEE effect differentially (azimuthal efficiency effect, which causes different V0 topologies to be enhanced/suppressed):
            histos.fill(HIST("FakePolDrivers/PhiAEE/h2dMassVsPhiAEE"), v0LambdaLikeMass, deltaPhiLambdaProtonStar);
            // AEE and HEE correlation:
            histos.fill(HIST("FakePolDrivers/CosHEEVsPhiStar/h2dFakePolCounts_CosThetaVsPhiStar"), cosFakePol, phiStar);
          } // end doFakePolDiagnosticsQA (per-V0 AEE/HEE)

          // Useful kinematic bools:
          const bool lambdaEtaPos = v0eta >= 0.;
          const bool pTLambdaCheck = v0pt > 0.5 && v0pt < 1.5;
          const bool rapidityLambdaCheck = std::abs(lambdaRapidity) < 0.5;
          const bool kinematicLambdaCheck = pTLambdaCheck && rapidityLambdaCheck; // Bins 1 and 3 of IntegratedCuts/ (hardcoded, may differ from analysisLevelCuts)

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
          const ProxyRingCache ringCacheLeadP = leadP.isValid ? computeProxyRing(leadPUnitVec, leadP.phi, v0RingInputs) : ProxyRingCache{};
          const ProxyRingCache ringCacheLeadJet = leadJet.isValid ? computeProxyRing(leadJetUnitVec, leadJet.phi, v0RingInputs) : ProxyRingCache{};
          const ProxyRingCache ringCacheSubJet = subJet.isValid ? computeProxyRing(subJetUnitVec, subJet.phi, v0RingInputs) : ProxyRingCache{};

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
              ringObservableOverLeadJetZ = ringCacheLeadJet.ringObservable / leadJetZ;
            else
              ringObservableOverLeadJetZ = 0.0; // A simple guard. May not be the best, but works
          }

          float v0phiToFillHists = wrapToPiFast(v0phi); // A short wrap to reuse some predefined axes

          ////////////////////
          // Fill helpers for this V0 -- Declared as lambdas, so they read the V0 variables above directly:
          // First, a helper for the ring observable histograms of one proxy.
          // -- One call per V0 and valid proxy.
          // -- Inputs:
          // proxy: the proxy actually used, after applyProxyDistortion.
          // proxyZ: \hat t_z of the proxy, for the RingKernel (LeadJet and LeadP only for now).
          // proxyEtaPos: eta_proxy >= 0, for the LeadP eta split.
          // -- Note:
          // The common block is what every proxy fills. The LeadJet also fills the whole QA set, the LeadP adds the RingMaps,
          // the RingKernel and the eta split, and the SubJet stays with the common block.
          auto fillRingObservable = [&](auto proxyTypeTag, const ProxyKinematics& proxy, float proxyZ, bool proxyEtaPos, const ProxyRingCache& ringCache) {
            // Gets the enum value at compile time via decltype's ::value, so HIST() can use it (a for loop over the enum would only give it at runtime):
            constexpr int idxProxy = decltype(proxyTypeTag)::value;
            constexpr auto folderPath = HIST("Ring/") + HIST(ProxyNames[idxProxy]);

            // Counters
            histosRing.fill(folderPath + HIST("/V0ProxyPairQA/hDeltaPhi"), ringCache.deltaPhi);
            histosRing.fill(folderPath + HIST("/V0ProxyPairQA/hDeltaTheta"), ringCache.deltaTheta);
            histosRing.fill(folderPath + HIST("/V0ProxyPairQA/hCosDeltaTheta"), ringCache.cosDeltaTheta);
            histosRing.fill(folderPath + HIST("/V0ProxyPairQA/hPairedProxyPt"), proxy.pt);
            // TProfile of Ring vs Mass
            histosRing.fill(folderPath + HIST("/MassDependence/pRingObservableMass"), v0LambdaLikeMass, ringCache.ringObservable);
            // Additional plots for instant gratification - 1D Profiles
            histosRing.fill(folderPath + HIST("/hRingObservableCounts"), ringCache.ringObservable);
            histosRing.fill(folderPath + HIST("/AngularDependence/pRingObservableDeltaPhi"), ringCache.deltaPhi, ringCache.ringObservable);
            histosRing.fill(folderPath + HIST("/AngularDependence/pRingObservableDeltaTheta"), ringCache.deltaTheta, ringCache.ringObservable);
            histosRing.fill(folderPath + HIST("/EtaDependence/pRingObservableEtaLambda"), v0eta, ringCache.ringObservable);
            histosRing.fill(folderPath + HIST("/EtaDependence/pRingObservableProxyEta"), proxy.eta, ringCache.ringObservable);
            histosRing.fill(folderPath + HIST("/pRingObservableIntegrated"), 0., ringCache.ringObservable);
            histosRing.fill(folderPath + HIST("/PtDependence/pRingObservableLambdaPt"), v0pt, ringCache.ringObservable);
            histosRing.fill(folderPath + HIST("/EventLevelDependence/pRingObservablePVz"), collisionPVz, ringCache.ringObservable);
            histosRing.fill(folderPath + HIST("/PtDependence/pRingVsProxyPt"), proxy.pt, ringCache.ringObservable);
            histosRing.fill(folderPath + HIST("/PtDependence/p2dRingVsProxyPtVsProxyEta"), proxy.pt, proxy.eta, ringCache.ringObservable);
            histosRing.fill(folderPath + HIST("/PtDependence/p2dRingVsProxyPtVsEtaV0"), proxy.pt, v0eta, ringCache.ringObservable);
            histosRing.fill(folderPath + HIST("/PtDependence/p2dRingVsProxyPtVsCentrality"), proxy.pt, centrality, ringCache.ringObservable);
            histosRing.fill(folderPath + HIST("/EtaDependence/p2dRingObservableEtaLambdaVsProxyEta"), v0eta, proxy.eta, ringCache.ringObservable);
            histosRing.fill(folderPath + HIST("/EtaDependence/h2dCounterEtaLambdaVsProxyEta"), v0eta, proxy.eta);
            // 2D Profiles: EtaProxy vs Mass
            histosRing.fill(folderPath + HIST("/MassDependence/p2dRingObservableProxyEtaVsMass"), proxy.eta, v0LambdaLikeMass, ringCache.ringObservable);
            histosRing.fill(folderPath + HIST("/MassDependence/h2dCounterProxyEtaVsMass"), proxy.eta, v0LambdaLikeMass);
            // KappaEff moments: kappa = 3 <num>/<den>, and products for the covariances
            histosRing.fill(folderPath + HIST("/KappaEff/pKappaNumVsMass"), v0LambdaLikeMass, ringCache.kappaNum);
            histosRing.fill(folderPath + HIST("/KappaEff/pKappaDenVsMass"), v0LambdaLikeMass, ringCache.kappaDen);
            histosRing.fill(folderPath + HIST("/KappaEff/pKappaNumTimesDenVsMass"), v0LambdaLikeMass, ringCache.kappaNum * ringCache.kappaDen);
            histosRing.fill(folderPath + HIST("/KappaEff/pRingTimesDenVsMass"), v0LambdaLikeMass, ringCache.ringObservable * ringCache.kappaDen);
            // CheapSigExtract mass region equivalents of KappaEff (bin 1 sideband, bin 2 peak, otherwise underflow)
            histosRing.fill(folderPath + HIST("/KappaEff/pRingVsMassRegion"), massRegion, ringCache.ringObservable);
            histosRing.fill(folderPath + HIST("/KappaEff/pKappaNumVsMassRegion"), massRegion, ringCache.kappaNum);
            histosRing.fill(folderPath + HIST("/KappaEff/pKappaDenVsMassRegion"), massRegion, ringCache.kappaDen);
            histosRing.fill(folderPath + HIST("/KappaEff/pKappaNumTimesDenVsMassRegion"), massRegion, ringCache.kappaNum * ringCache.kappaDen);
            histosRing.fill(folderPath + HIST("/KappaEff/pRingTimesDenVsMassRegion"), massRegion, ringCache.ringObservable * ringCache.kappaDen);
            // Differential KappaEff -- vs proxy pT and vs centrality, still per mass region (axes shared with event mixing)
            histosRing.fill(folderPath + HIST("/KappaEff/Differential/p2dRingVsProxyPtVsMassRegion"), proxy.pt, massRegion, ringCache.ringObservable);
            histosRing.fill(folderPath + HIST("/KappaEff/Differential/p2dKappaNumVsProxyPtVsMassRegion"), proxy.pt, massRegion, ringCache.kappaNum);
            histosRing.fill(folderPath + HIST("/KappaEff/Differential/p2dKappaDenVsProxyPtVsMassRegion"), proxy.pt, massRegion, ringCache.kappaDen);
            histosRing.fill(folderPath + HIST("/KappaEff/Differential/p2dKappaNumTimesDenVsProxyPtVsMassRegion"), proxy.pt, massRegion, ringCache.kappaNum * ringCache.kappaDen);
            histosRing.fill(folderPath + HIST("/KappaEff/Differential/p2dRingTimesDenVsProxyPtVsMassRegion"), proxy.pt, massRegion, ringCache.ringObservable * ringCache.kappaDen);
            histosRing.fill(folderPath + HIST("/KappaEff/Differential/p2dRingVsCentralityVsMassRegion"), centrality, massRegion, ringCache.ringObservable);
            histosRing.fill(folderPath + HIST("/KappaEff/Differential/p2dKappaNumVsCentralityVsMassRegion"), centrality, massRegion, ringCache.kappaNum);
            histosRing.fill(folderPath + HIST("/KappaEff/Differential/p2dKappaDenVsCentralityVsMassRegion"), centrality, massRegion, ringCache.kappaDen);
            histosRing.fill(folderPath + HIST("/KappaEff/Differential/p2dKappaNumTimesDenVsCentralityVsMassRegion"), centrality, massRegion, ringCache.kappaNum * ringCache.kappaDen);
            histosRing.fill(folderPath + HIST("/KappaEff/Differential/p2dRingTimesDenVsCentralityVsMassRegion"), centrality, massRegion, ringCache.ringObservable * ringCache.kappaDen);

            // Leading jet and leading particle:
            if constexpr (idxProxy == kLeadJet || idxProxy == kLeadP) {
              histosRing.fill(folderPath + HIST("/EtaDependence/pRingObservableProxyEtaHighEtaRes"), proxy.eta, ringCache.ringObservable);
              // 2D Profiles of the ring: lab momentum planes
              histosRing.fill(folderPath + HIST("/RingMaps/p2dRingObservableVsPxPy"), v0px, v0py, ringCache.ringObservable);
              histosRing.fill(folderPath + HIST("/RingMaps/p2dRingObservableVsPzPx"), v0pz, v0px, ringCache.ringObservable);
              histosRing.fill(folderPath + HIST("/RingMaps/p2dRingObservableVsPyPz"), v0py, v0pz, ringCache.ringObservable);
              // Ring projection kernel
              histosRing.fill(folderPath + HIST("/RingKernel/p2dRingObservableCosDeltaThetaVsProxyZ"), ringCache.cosDeltaTheta, proxyZ, ringCache.ringObservable);
              histosRing.fill(folderPath + HIST("/RingKernel/h2dCountsCosDeltaThetaVsProxyZ"), ringCache.cosDeltaTheta, proxyZ);
              histosRing.fill(folderPath + HIST("/RingKernel/p3dRingObservableCosDeltaThetaVsProxyZVsLambdaZ"), ringCache.cosDeltaTheta, proxyZ, lambdaZ, ringCache.ringObservable);
            }

            // Leading particle only -- all eta checks for leading particle and V0s, along with the fills:
            if constexpr (idxProxy == kLeadP) {
              if (proxyEtaPos) {
                // leadP marginal: positive side
                histosRing.fill(folderPath + HIST("/PtDependence/pRingVsProxyPt_PosProxyEta"), proxy.pt, ringCache.ringObservable);
                if (lambdaEtaPos) {
                  // V0 marginal: positive side
                  histosRing.fill(folderPath + HIST("/PtDependence/pRingVsProxyPt_PosEtaV0"), proxy.pt, ringCache.ringObservable);
                  // Joint: (+leadP, +V0)
                  histosRing.fill(folderPath + HIST("/PtDependence/pRingVsProxyPt_PosProxyEta_PosEtaV0"), proxy.pt, ringCache.ringObservable);
                } else {
                  // V0 marginal: negative side
                  histosRing.fill(folderPath + HIST("/PtDependence/pRingVsProxyPt_NegEtaV0"), proxy.pt, ringCache.ringObservable);
                  // Joint: (+leadP, -V0)
                  histosRing.fill(folderPath + HIST("/PtDependence/pRingVsProxyPt_PosProxyEta_NegEtaV0"), proxy.pt, ringCache.ringObservable);
                }
              } else {
                // leadP marginal: negative side
                histosRing.fill(folderPath + HIST("/PtDependence/pRingVsProxyPt_NegProxyEta"), proxy.pt, ringCache.ringObservable);
                if (lambdaEtaPos) {
                  // V0 marginal: positive side
                  histosRing.fill(folderPath + HIST("/PtDependence/pRingVsProxyPt_PosEtaV0"), proxy.pt, ringCache.ringObservable);
                  // Joint: (-leadP, +V0)
                  histosRing.fill(folderPath + HIST("/PtDependence/pRingVsProxyPt_NegProxyEta_PosEtaV0"), proxy.pt, ringCache.ringObservable);
                } else {
                  // V0 marginal: negative side
                  histosRing.fill(folderPath + HIST("/PtDependence/pRingVsProxyPt_NegEtaV0"), proxy.pt, ringCache.ringObservable);
                  // Joint: (-leadP, -V0)
                  histosRing.fill(folderPath + HIST("/PtDependence/pRingVsProxyPt_NegProxyEta_NegEtaV0"), proxy.pt, ringCache.ringObservable);
                }
              }
            }

            // Leading jet only:
            if constexpr (idxProxy == kLeadJet) {
              // Counters
              histosRing.fill(folderPath + HIST("/V0ProxyPairQA/h2dDeltaPhiVsDeltaEta"), ringCache.deltaPhi, deltaEtaLeadJet);
              histosRing.fill(folderPath + HIST("/hIntegrated"), 0.);
              histosRing.fill(folderPath + HIST("/V0ProxyPairQA/hPairedV0Pt"), v0pt);
              // Counters
              histosRing.fill(folderPath + HIST("/PtDependence/h2dDeltaPhiVsLambdaPt"), ringCache.deltaPhi, v0pt);
              histosRing.fill(folderPath + HIST("/PtDependence/h2dDeltaThetaVsLambdaPt"), ringCache.deltaTheta, v0pt);
              histosRing.fill(folderPath + HIST("/V0ProxyPairQA/h2dDeltaPhiVsProxyPhi"), ringCache.deltaPhi, proxy.phi);
              // Additional plots for instant gratification - 1D Profiles
              histosRing.fill(folderPath + HIST("/AngularDependence/pRingObservableProxyPhi"), proxy.phi, ringCache.ringObservable);
              histosRing.fill(folderPath + HIST("/AngularDependence/pRingObservablePhiLambda"), v0phi, ringCache.ringObservable);
              // 2D Profiles
              histosRing.fill(folderPath + HIST("/PtDependence/p2dRingObservableDeltaPhiVsLambdaPt"), ringCache.deltaPhi, v0pt, ringCache.ringObservable);
              histosRing.fill(folderPath + HIST("/PtDependence/p2dRingObservableDeltaThetaVsLambdaPt"), ringCache.deltaTheta, v0pt, ringCache.ringObservable);
              histosRing.fill(folderPath + HIST("/PtDependence/p2dRingObservableDeltaPhiVsProxyPt"), ringCache.deltaPhi, proxy.pt, ringCache.ringObservable);
              histosRing.fill(folderPath + HIST("/PtDependence/p2dRingObservableDeltaThetaVsProxyPt"), ringCache.deltaTheta, proxy.pt, ringCache.ringObservable);
              // 1D Mass
              histosRing.fill(folderPath + HIST("/V0ProxyPairQA/hPairedV0Mass"), v0LambdaLikeMass);
              histosRing.fill(folderPath + HIST("/MassDependence/hMassSigExtract"), v0LambdaLikeMass);
              // Counters
              histosRing.fill(folderPath + HIST("/MassDependence/h2dDeltaPhiVsMass"), ringCache.deltaPhi, v0LambdaLikeMass);
              histosRing.fill(folderPath + HIST("/MassDependence/h2dDeltaThetaVsMass"), ringCache.deltaTheta, v0LambdaLikeMass);
              // 2D Profiles: Angle vs Mass
              histosRing.fill(folderPath + HIST("/MassDependence/p2dRingObservableDeltaPhiVsMass"), ringCache.deltaPhi, v0LambdaLikeMass, ringCache.ringObservable);
              histosRing.fill(folderPath + HIST("/MassDependence/p2dRingObservableDeltaThetaVsMass"), ringCache.deltaTheta, v0LambdaLikeMass, ringCache.ringObservable);
              histosRing.fill(folderPath + HIST("/MassDependence/p2dRingObservableEtaLambdaVsMass"), v0eta, v0LambdaLikeMass, ringCache.ringObservable);
              // 2D Profile: Ring vs Eta variables
              histosRing.fill(folderPath + HIST("/EtaDependence/hCounterEtaLambdaMinusProxyEta"), v0eta - proxy.eta);
              histosRing.fill(folderPath + HIST("/EtaDependence/pRingObservableEtaLambdaMinusProxyEta"), v0eta - proxy.eta, ringCache.ringObservable);
              histosRing.fill(folderPath + HIST("/EtaDependence/p2dRingObservableEtaLambdaVsProxyEta_FineBins"), v0eta, proxy.eta, ringCache.ringObservable);
              histosRing.fill(folderPath + HIST("/EtaDependence/h2dCounterEtaLambdaVsProxyEta_FineBins"), v0eta, proxy.eta);
              // 3D Profiles: Angle vs Mass vs Lambda pT
              histosRing.fill(folderPath + HIST("/SigExtract3DInput/p3dRingObservableDeltaPhiVsMassVsLambdaPt"), ringCache.deltaPhi, v0LambdaLikeMass, v0pt, ringCache.ringObservable);
              histosRing.fill(folderPath + HIST("/SigExtract3DInput/p3dRingObservableDeltaThetaVsMassVsLambdaPt"), ringCache.deltaTheta, v0LambdaLikeMass, v0pt, ringCache.ringObservable);
              // 3D Profiles: Angle vs Mass vs Lead Jet pT
              histosRing.fill(folderPath + HIST("/SigExtract3DInput/p3dRingObservableDeltaPhiVsMassVsProxyPt"), ringCache.deltaPhi, v0LambdaLikeMass, proxy.pt, ringCache.ringObservable);
              histosRing.fill(folderPath + HIST("/SigExtract3DInput/p3dRingObservableDeltaThetaVsMassVsProxyPt"), ringCache.deltaTheta, v0LambdaLikeMass, proxy.pt, ringCache.ringObservable);
              // 2D Profile: Mass vs Centrality
              histosRing.fill(folderPath + HIST("/MassDependence/p2dRingObservableMassVsCent"), v0LambdaLikeMass, centrality, ringCache.ringObservable);
              // 3D Profiles: Angle vs Mass vs Centrality
              histosRing.fill(folderPath + HIST("/SigExtract3DInput/p3dRingObservableDeltaPhiVsMassVsCent"), ringCache.deltaPhi, v0LambdaLikeMass, centrality, ringCache.ringObservable);
              histosRing.fill(folderPath + HIST("/SigExtract3DInput/p3dRingObservableDeltaThetaVsMassVsCent"), ringCache.deltaTheta, v0LambdaLikeMass, centrality, ringCache.ringObservable);
              histosRing.fill(folderPath + HIST("/EventLevelDependence/pRingVsCentrality"), centrality, ringCache.ringObservable);
              // (TODO: add counters for regular TH2Ds about centrality)
              // Ring scalar on the AEE planes. R is invariant under the rotation, so only the binning should change
              histosRing.fill(folderPath + HIST("/RingMaps/p2dRingObservableVsPxAeePyAee"), v0pxAee, v0pyAee, ringCache.ringObservable);
              histosRing.fill(folderPath + HIST("/RingMaps/p2dRingObservableVsPzPxAee"), v0pz, v0pxAee, ringCache.ringObservable);
              histosRing.fill(folderPath + HIST("/RingMaps/p2dRingObservableVsPyAeePz"), v0pyAee, v0pz, ringCache.ringObservable);
              // Ring scalar on the coordinates transverse to the jet (PrimeJet).
              histosRing.fill(folderPath + HIST("/RingMaps/p2dRingObservableVsPxPyPrimeJet"), v0pxPrimeJet, v0pyPrimeJet, ringCache.ringObservable);
              histosRing.fill(folderPath + HIST("/RingMaps/p2dRingObservableVsPzPxPrimeJet"), v0pzPrimeJet, v0pxPrimeJet, ringCache.ringObservable);
              histosRing.fill(folderPath + HIST("/RingMaps/p2dRingObservableVsPyPzPrimeJet"), v0pyPrimeJet, v0pzPrimeJet, ringCache.ringObservable);
              // Jet-transverse coordinate system's polarization maps. Filled here to use existing leadJet.isValid check
              histosRing.fill(HIST("RawPolarization/PolMaps/PrimeJet/p2dPxStarPrimeJet_vsPxPyPrimeJet"), v0pxPrimeJet, v0pyPrimeJet, polStarXPrimeJet);
              histosRing.fill(HIST("RawPolarization/PolMaps/PrimeJet/p2dPyStarPrimeJet_vsPxPyPrimeJet"), v0pxPrimeJet, v0pyPrimeJet, polStarYPrimeJet);
              histosRing.fill(HIST("RawPolarization/PolMaps/PrimeJet/p2dPzStarPrimeJet_vsPxPyPrimeJet"), v0pxPrimeJet, v0pyPrimeJet, polStarZPrimeJet);
              histosRing.fill(HIST("RawPolarization/PolMaps/PrimeJet/h2dCountsVsPxPyPrimeJet"), v0pxPrimeJet, v0pyPrimeJet);
              histosRing.fill(HIST("RawPolarization/PolMaps/PrimeJet/p2dPxStarPrimeJet_vsPzPxPrimeJet"), v0pzPrimeJet, v0pxPrimeJet, polStarXPrimeJet);
              histosRing.fill(HIST("RawPolarization/PolMaps/PrimeJet/p2dPyStarPrimeJet_vsPzPxPrimeJet"), v0pzPrimeJet, v0pxPrimeJet, polStarYPrimeJet);
              histosRing.fill(HIST("RawPolarization/PolMaps/PrimeJet/p2dPzStarPrimeJet_vsPzPxPrimeJet"), v0pzPrimeJet, v0pxPrimeJet, polStarZPrimeJet);
              histosRing.fill(HIST("RawPolarization/PolMaps/PrimeJet/h2dCountsVsPzPxPrimeJet"), v0pzPrimeJet, v0pxPrimeJet);
              histosRing.fill(HIST("RawPolarization/PolMaps/PrimeJet/p2dPxStarPrimeJet_vsPyPzPrimeJet"), v0pyPrimeJet, v0pzPrimeJet, polStarXPrimeJet);
              histosRing.fill(HIST("RawPolarization/PolMaps/PrimeJet/p2dPyStarPrimeJet_vsPyPzPrimeJet"), v0pyPrimeJet, v0pzPrimeJet, polStarYPrimeJet);
              histosRing.fill(HIST("RawPolarization/PolMaps/PrimeJet/p2dPzStarPrimeJet_vsPyPzPrimeJet"), v0pyPrimeJet, v0pzPrimeJet, polStarZPrimeJet);
              histosRing.fill(HIST("RawPolarization/PolMaps/PrimeJet/h2dCountsVsPyPzPrimeJet"), v0pyPrimeJet, v0pzPrimeJet);
              // QAs on DeltaPhiJet -- Filled here, as the LeadJet ringCache.deltaPhi is only defined under leadJet.isValid
              constexpr auto rawPolarizationFolderPath = HIST("RawPolarization/") + HIST(ProxyNames[idxProxy]);
              histosRing.fill(rawPolarizationFolderPath + HIST("/pPxStarDeltaPhi"), ringCache.deltaPhi, polStarX);
              histosRing.fill(rawPolarizationFolderPath + HIST("/pPyStarDeltaPhi"), ringCache.deltaPhi, polStarY);
              histosRing.fill(rawPolarizationFolderPath + HIST("/pPzStarDeltaPhi"), ringCache.deltaPhi, polStarZ);
              histosRing.fill(rawPolarizationFolderPath + HIST("/p2dPxStarDeltaPhiVsLambdaPt"), ringCache.deltaPhi, v0pt, polStarX);
              histosRing.fill(rawPolarizationFolderPath + HIST("/p2dPyStarDeltaPhiVsLambdaPt"), ringCache.deltaPhi, v0pt, polStarY);
              histosRing.fill(rawPolarizationFolderPath + HIST("/p2dPzStarDeltaPhiVsLambdaPt"), ringCache.deltaPhi, v0pt, polStarZ);
            }
          };

          // The IntegratedCuts/<Proxy>/ histogram filler, by hand:
          // -- Bins (as booked, i.e. the old kinematic-cut families): 0 all Lambdas, 1 Lambda pT and rapidity cuts, 2 proxy |eta| < 0.5, 3 both.
          // -- Inputs:
          // proxyKinematicCheck: |eta_proxy| < 0.5, for bins 2 and 3.
          auto fillIntegratedCuts = [&](auto proxyTypeTag, bool proxyKinematicCheck, const ProxyRingCache& ringCache) {
            // Gets the enum value at compile time via decltype's ::value, so HIST() can use it (a for loop over the enum would only give it at runtime):
            constexpr int idxProxy = decltype(proxyTypeTag)::value;
            constexpr auto folderPath = HIST("IntegratedCuts/") + HIST(ProxyNames[idxProxy]);
            // One entry per bin, true when the V0 (and this proxy) pass that bin's cuts:
            const std::array<bool, 4> passedCuts = {true, kinematicLambdaCheck, proxyKinematicCheck, kinematicLambdaCheck && proxyKinematicCheck};
            for (int cutsBin = 0; cutsBin < static_cast<int>(passedCuts.size()); cutsBin++) {
              if (!passedCuts[cutsBin])
                continue;
              histos.fill(folderPath + HIST("/pRingCuts"), cutsBin, ringCache.ringObservable);
              if (v0InMassPeak)
                histos.fill(folderPath + HIST("/p2dRingCutsV0MassPeak"), cutsBin, 1, ringCache.ringObservable); // Fills the inPeak bin
              else if (v0InMassWindow)
                histos.fill(folderPath + HIST("/p2dRingCutsV0MassPeak"), cutsBin, 0, ringCache.ringObservable); // Fills the inWindow bin, which has a stricter interval than !v0InMassPeak
              histos.fill(folderPath + HIST("/hCountCuts"), cutsBin);
            }
          };

          // The helper for longitudinal-ring diagnostics of one proxy (R_z and R_z^T only).
          // One call per V0 and valid proxy.
          auto fillRzDiagnostics = [&](auto proxyTypeTag, const ProxyRingCache& ringCache) {
            // Gets the enum value at compile time via decltype's ::value, so HIST() can use it (a for loop over the enum would only give it at runtime):
            constexpr int idxProxy = decltype(proxyTypeTag)::value;
            constexpr auto folderPath = HIST("RzDiagnostics/") + HIST(ProxyNames[idxProxy]);
            histos.fill(HIST("RzDiagnostics/pHelicityTermIntegrated"), idxProxy + 0.5, ringCache.helicityTerm); // One bin per proxy (see ProxyTypeIndex)
            histos.fill(folderPath + HIST("/pHelicityTermVsEtaLambda"), v0eta, ringCache.helicityTerm);
            histos.fill(folderPath + HIST("/pNzVsEtaLambda"), v0eta, ringCache.nz);
            histos.fill(folderPath + HIST("/pNzSqVsEtaLambda"), v0eta, ringCache.nz * ringCache.nz);
            histos.fill(folderPath + HIST("/pRingVsChi"), ringCache.chi, ringCache.ringObservable);
            histos.fill(folderPath + HIST("/p2dRingVsChiVsMass"), ringCache.chi, v0LambdaLikeMass, ringCache.ringObservable);
            histos.fill(HIST("RzDiagnostics/pRingPerpIntegrated"), idxProxy + 0.5, ringCache.ringPerp);
            histos.fill(folderPath + HIST("/pRingPerpVsMass"), v0LambdaLikeMass, ringCache.ringPerp);
            histos.fill(folderPath + HIST("/pRingPerpVsPhiAEE"), deltaPhiLambdaProtonStar, ringCache.ringPerp);
            histos.fill(folderPath + HIST("/pRingPerpVsDeltaPhi"), ringCache.deltaPhi, ringCache.ringPerp);
            if constexpr (idxProxy == kLeadJet) {
              histos.fill(folderPath + HIST("/p2dHelicityTermVsDeltaPhiVsMass"), ringCache.deltaPhi, v0LambdaLikeMass, ringCache.helicityTerm);
              if (v0eta > 0.f)
                histos.fill(folderPath + HIST("/p2dRingVsDeltaPhiVsMassEtaPos"), ringCache.deltaPhi, v0LambdaLikeMass, ringCache.ringObservable);
              else
                histos.fill(folderPath + HIST("/p2dRingVsDeltaPhiVsMassEtaNeg"), ringCache.deltaPhi, v0LambdaLikeMass, ringCache.ringObservable);
              histos.fill(folderPath + HIST("/p2dRingVsDeltaPhiVsDeltaEta"), ringCache.deltaPhi, deltaEtaLeadJet, ringCache.ringObservable);
              histos.fill(folderPath + HIST("/pCosSqThetaStarZVsMass"), v0LambdaLikeMass, protonLikeStarUnit3Vec.Z() * protonLikeStarUnit3Vec.Z());
            }
          };

          // The per-proxy part of the doFakePolDiagnosticsQA block: FakePolDrivers/{ProxyEta, CosHEE, PhiAEE, PhiStar}/<Proxy>/.
          // One call per V0 and valid proxy.
          // -- Inputs:
          // proxyEtaPos: eta_proxy >= 0.
          // proxyEtaStrict: |eta_proxy| >= R (jets only: the LeadP axis stops at 9 bins, so it has no strict variant).
          auto fillFakePolDiagnostics = [&](auto proxyTypeTag, const ProxyKinematics& proxy, bool proxyEtaPos, bool proxyEtaStrict, const ProxyRingCache& ringCache) {
            // Gets the enum value at compile time via decltype's ::value, so HIST() can use it (a for loop over the enum would only give it at runtime):
            constexpr int idxProxy = decltype(proxyTypeTag)::value;
            constexpr auto proxyEtaFolderPath = HIST("FakePolDrivers/ProxyEta/") + HIST(ProxyNames[idxProxy]);
            constexpr auto phiAeeFolderPath = HIST("FakePolDrivers/PhiAEE/") + HIST(ProxyNames[idxProxy]);

            // Filling eta dependence QAs of the result (both for V0 and jet proxy):
            // Bin 0: all, 1/2: proxy #eta sign, 3/4: #Lambda #eta sign, 5-8: combinations of proxy and Lambda eta signs,
            // 9/10: |#eta_{proxy}| >= R, 11-14: "stricter" combinations based on bins 9/10.
            const int etaLambdaBin = lambdaEtaPos ? 3 : 4; // Depends on the V0 only
            const int etaProxyBin = proxyEtaPos ? 1 : 2;
            const int etaProxyLambdaBin = (proxyEtaPos ? 5 : 7) + (lambdaEtaPos ? 0 : 1);
            histos.fill(proxyEtaFolderPath + HIST("/pRingEtaCuts"), 0, ringCache.ringObservable);
            histos.fill(proxyEtaFolderPath + HIST("/pRingEtaCuts"), etaLambdaBin, ringCache.ringObservable);
            histos.fill(proxyEtaFolderPath + HIST("/pRingEtaCuts"), etaProxyBin, ringCache.ringObservable);
            histos.fill(proxyEtaFolderPath + HIST("/pRingEtaCuts"), etaProxyLambdaBin, ringCache.ringObservable);
            if constexpr (idxProxy != kLeadP) {
              if (proxyEtaStrict) { // |eta_{proxy}| >= R
                const int etaProxyStrictBin = proxyEtaPos ? 9 : 10;
                const int etaProxyStrictLambdaBin = (proxyEtaPos ? 11 : 13) + (lambdaEtaPos ? 0 : 1);
                histos.fill(proxyEtaFolderPath + HIST("/pRingEtaCuts"), etaProxyStrictBin, ringCache.ringObservable);
                histos.fill(proxyEtaFolderPath + HIST("/pRingEtaCuts"), etaProxyStrictLambdaBin, ringCache.ringObservable);
              }
            }

            // Studying the signal Vs background integral (a very naive estimative of the invariant mass peak):
            if constexpr (idxProxy == kLeadP) {
              if (v0InMassPeak) {
                histos.fill(proxyEtaFolderPath + HIST("/p2dRingEtaCuts_MassSignalVsBackground"), 0, 1, ringCache.ringObservable);
                histos.fill(proxyEtaFolderPath + HIST("/p2dRingEtaCuts_MassSignalVsBackground"), etaLambdaBin, 1, ringCache.ringObservable);
                histos.fill(proxyEtaFolderPath + HIST("/p2dRingEtaCuts_MassSignalVsBackground"), etaProxyBin, 1, ringCache.ringObservable);
                histos.fill(proxyEtaFolderPath + HIST("/p2dRingEtaCuts_MassSignalVsBackground"), etaProxyLambdaBin, 1, ringCache.ringObservable);
              } else if (v0InMassWindow) {
                histos.fill(proxyEtaFolderPath + HIST("/p2dRingEtaCuts_MassSignalVsBackground"), 0, 0, ringCache.ringObservable);
                histos.fill(proxyEtaFolderPath + HIST("/p2dRingEtaCuts_MassSignalVsBackground"), etaLambdaBin, 0, ringCache.ringObservable);
                histos.fill(proxyEtaFolderPath + HIST("/p2dRingEtaCuts_MassSignalVsBackground"), etaProxyBin, 0, ringCache.ringObservable);
                histos.fill(proxyEtaFolderPath + HIST("/p2dRingEtaCuts_MassSignalVsBackground"), etaProxyLambdaBin, 0, ringCache.ringObservable);
              }
            }

            // HEE study (helicity efficiency effect), for the leading jet and the leading particle:
            if constexpr (idxProxy == kLeadJet || idxProxy == kLeadP) {
              constexpr auto cosHeeFolderPath = HIST("FakePolDrivers/CosHEE/") + HIST(ProxyNames[idxProxy]);
              histos.fill(cosHeeFolderPath + HIST("/h2dFakePolCounts"), cosFakePol, 0);
              histos.fill(cosHeeFolderPath + HIST("/h2dFakePolCounts"), cosFakePol, etaLambdaBin);
              histos.fill(cosHeeFolderPath + HIST("/h2dFakePolCounts"), cosFakePol, etaProxyBin);
              histos.fill(cosHeeFolderPath + HIST("/h2dFakePolCounts"), cosFakePol, etaProxyLambdaBin);

              // Extra correlations test:
              histos.fill(cosHeeFolderPath + HIST("/h2dFakePolCountsVsDeltaTheta"), cosFakePol, ringCache.deltaTheta);
              histos.fill(cosHeeFolderPath + HIST("/h2dFakePolCountsCosThetaVsLambdaPt"), cosFakePol, v0pt); // For the LeadP: understanding the population of events that has a leading particle (even though this does not need one to be calculated!)
              // Split by proxy #eta sign:
              if (proxyEtaPos)
                histos.fill(cosHeeFolderPath + HIST("/h2dFakePolCountsVsDeltaThetaPosProxyEta"), cosFakePol, ringCache.deltaTheta);
              else
                histos.fill(cosHeeFolderPath + HIST("/h2dFakePolCountsVsDeltaThetaNegProxyEta"), cosFakePol, ringCache.deltaTheta);
            }

            if constexpr (idxProxy == kLeadJet) {
              constexpr auto cosHeeFolderPath = HIST("FakePolDrivers/CosHEE/") + HIST(ProxyNames[idxProxy]);
              constexpr auto phiStarFolderPath = HIST("FakePolDrivers/PhiStar/") + HIST(ProxyNames[idxProxy]);
              // Same for signal:
              histos.fill(cosHeeFolderPath + HIST("/p2dFakePolSignalVsCosTheta"), cosFakePol, 0, ringCache.ringObservable);
              histos.fill(cosHeeFolderPath + HIST("/p2dFakePolSignalVsCosTheta"), cosFakePol, etaLambdaBin, ringCache.ringObservable);
              histos.fill(cosHeeFolderPath + HIST("/p2dFakePolSignalVsCosTheta"), cosFakePol, etaProxyBin, ringCache.ringObservable);
              histos.fill(cosHeeFolderPath + HIST("/p2dFakePolSignalVsCosTheta"), cosFakePol, etaProxyLambdaBin, ringCache.ringObservable);
              // Counter and ring accumulators for AEE study:
              histos.fill(phiStarFolderPath + HIST("/h2dCountsVsPhiStar"), phiStar, 0);
              histos.fill(phiStarFolderPath + HIST("/h2dCountsVsPhiStar"), phiStar, etaLambdaBin);
              histos.fill(phiStarFolderPath + HIST("/h2dCountsVsPhiStar"), phiStar, etaProxyBin);
              histos.fill(phiStarFolderPath + HIST("/h2dCountsVsPhiStar"), phiStar, etaProxyLambdaBin);
              histos.fill(phiStarFolderPath + HIST("/p2dFakePolSignalVsPhiStar"), phiStar, 0, ringCache.ringObservable);
              histos.fill(phiStarFolderPath + HIST("/p2dFakePolSignalVsPhiStar"), phiStar, etaLambdaBin, ringCache.ringObservable);
              histos.fill(phiStarFolderPath + HIST("/p2dFakePolSignalVsPhiStar"), phiStar, etaProxyBin, ringCache.ringObservable);
              histos.fill(phiStarFolderPath + HIST("/p2dFakePolSignalVsPhiStar"), phiStar, etaProxyLambdaBin, ringCache.ringObservable);

              histos.fill(phiAeeFolderPath + HIST("/p2dFakePolSignalVsPhiAEE"), deltaPhiLambdaProtonStar, 0, ringCache.ringObservable);
              histos.fill(phiAeeFolderPath + HIST("/p2dFakePolSignalVsPhiAEE"), deltaPhiLambdaProtonStar, etaLambdaBin, ringCache.ringObservable);
              histos.fill(phiAeeFolderPath + HIST("/p2dFakePolSignalVsPhiAEE"), deltaPhiLambdaProtonStar, etaProxyBin, ringCache.ringObservable);
              histos.fill(phiAeeFolderPath + HIST("/p2dFakePolSignalVsPhiAEE"), deltaPhiLambdaProtonStar, etaProxyLambdaBin, ringCache.ringObservable);

              if (pTLambdaCheck) {
                histos.fill(cosHeeFolderPath + HIST("/h2dFakePolCountsLambdaPtCut"), cosFakePol, 0);
                histos.fill(cosHeeFolderPath + HIST("/h2dFakePolCountsLambdaPtCut"), cosFakePol, etaLambdaBin);
                histos.fill(cosHeeFolderPath + HIST("/h2dFakePolCountsLambdaPtCut"), cosFakePol, etaProxyBin);
                histos.fill(cosHeeFolderPath + HIST("/h2dFakePolCountsLambdaPtCut"), cosFakePol, etaProxyLambdaBin);
                if (rapidityLambdaCheck) { // Stricter check
                  histos.fill(cosHeeFolderPath + HIST("/h2dFakePolCountsLambdaPtYCuts"), cosFakePol, 0);
                  histos.fill(cosHeeFolderPath + HIST("/h2dFakePolCountsLambdaPtYCuts"), cosFakePol, etaLambdaBin);
                  histos.fill(cosHeeFolderPath + HIST("/h2dFakePolCountsLambdaPtYCuts"), cosFakePol, etaProxyBin);
                  histos.fill(cosHeeFolderPath + HIST("/h2dFakePolCountsLambdaPtYCuts"), cosFakePol, etaProxyLambdaBin);
                }
              }
            }

            // Dependencies for signal extraction (AEE, species-agnostic, see the bookings):
            histos.fill(phiAeeFolderPath + HIST("/pRingObservableVsPhiAEE"), deltaPhiLambdaProtonStar, ringCache.ringObservable);
            histos.fill(phiAeeFolderPath + HIST("/p2dRingObservableVsPhiAEEVsMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringCache.ringObservable);
            histos.fill(phiAeeFolderPath + HIST("/p2dRingObservableVsPhiAEEVsProxyEta"), deltaPhiLambdaProtonStar, proxy.eta, ringCache.ringObservable);
            histos.fill(phiAeeFolderPath + HIST("/p3dRingObservableVsPhiAEEVsProxyEtaVsMass"), deltaPhiLambdaProtonStar, proxy.eta, v0LambdaLikeMass, ringCache.ringObservable);
            if (proxyEtaPos) {
              histos.fill(phiAeeFolderPath + HIST("/p2dRingObservableVsPhiAEEVsMassPosProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringCache.ringObservable);
              histos.fill(phiAeeFolderPath + HIST("/p2dRingObservableVsPhiAEEVsProxyEtaSign"), deltaPhiLambdaProtonStar, 0.5, ringCache.ringObservable);
              if constexpr (idxProxy == kLeadJet) {
                // Background stabilization using etaLambda differentially:
                if (deltaPhiLambdaProtonStar >= 0) {
                  histos.fill(HIST("FakePolDrivers/PhiAEE/BkgStabilizationAttempt/p2dRingObservableEtaLambdaVsMassEtaProxyPosPhiAEEPos"), v0eta, v0LambdaLikeMass, ringCache.ringObservable);
                  if (ringCache.deltaPhi >= 0)
                    histos.fill(HIST("FakePolDrivers/PhiAEE/BkgStabilizationAttempt/p2dRingObservableEtaLambdaVsMassEtaProxyPosPhiAEEPosDeltaPhiLeadJetPos"), v0eta, v0LambdaLikeMass, ringCache.ringObservable);
                  else
                    histos.fill(HIST("FakePolDrivers/PhiAEE/BkgStabilizationAttempt/p2dRingObservableEtaLambdaVsMassEtaProxyPosPhiAEEPosDeltaPhiLeadJetNeg"), v0eta, v0LambdaLikeMass, ringCache.ringObservable);
                }
                // Background stabilization attempt with etaLambda sign only, PhiAEE differentially:
                if (v0eta >= 0) {
                  histos.fill(HIST("FakePolDrivers/PhiAEE/BkgStabilizationAttempt/p2dRingObservablePhiAEEVsMassEtaProxyPosPhiEtaLambdaPos"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringCache.ringObservable);
                  if (ringCache.deltaPhi >= 0)
                    histos.fill(HIST("FakePolDrivers/PhiAEE/BkgStabilizationAttempt/p2dRingObservablePhiAEEVsMassEtaProxyPosPhiEtaLambdaPosDeltaPhiLeadJetPos"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringCache.ringObservable);
                  else
                    histos.fill(HIST("FakePolDrivers/PhiAEE/BkgStabilizationAttempt/p2dRingObservablePhiAEEVsMassEtaProxyPosPhiEtaLambdaPosDeltaPhiLeadJetNeg"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringCache.ringObservable);
                }
              }
            } else {
              histos.fill(phiAeeFolderPath + HIST("/p2dRingObservableVsPhiAEEVsMassNegProxyEta"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringCache.ringObservable);
              histos.fill(phiAeeFolderPath + HIST("/p2dRingObservableVsPhiAEEVsProxyEtaSign"), deltaPhiLambdaProtonStar, -0.5, ringCache.ringObservable);
              if constexpr (idxProxy == kLeadJet) {
                // Background stabilization using etaLambda differentially:
                if (deltaPhiLambdaProtonStar >= 0)
                  histos.fill(HIST("FakePolDrivers/PhiAEE/BkgStabilizationAttempt/p2dRingObservableEtaLambdaVsMassEtaProxyNegPhiAEEPos"), v0eta, v0LambdaLikeMass, ringCache.ringObservable);
                // Background stabilization attempt with etaLambda sign only, PhiAEE differentially:
                if (v0eta >= 0)
                  histos.fill(HIST("FakePolDrivers/PhiAEE/BkgStabilizationAttempt/p2dRingObservablePhiAEEVsMassEtaProxyNegPhiEtaLambdaPos"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringCache.ringObservable);
              }
            }
            histos.fill(HIST("FakePolDrivers/PhiAEE/FiveBinMass/") + HIST(ProxyNames[idxProxy]) + HIST("/p2dRingObservableVsPhiAEEVs5BinMass"), deltaPhiLambdaProtonStar, v0LambdaLikeMass, ringCache.ringObservable);
          };
          // end of Fill helpers for this V0
          ////////////////////

          // Longitudinal-ring diagnostics (every proxy's ringObservable already holds R_z or R_z^T here):
          if (longitudinalRingMode) {
            if (leadJet.isValid)
              fillRzDiagnostics(ProxyTypeTag<kLeadJet>, ringCacheLeadJet);
            if (leadP.isValid)
              fillRzDiagnostics(ProxyTypeTag<kLeadP>, ringCacheLeadP);
            if (subJet.isValid)
              fillRzDiagnostics(ProxyTypeTag<kSubJet>, ringCacheSubJet);
          }

          // Fill ring histograms: (1D, lambda 2D correlations and jet 2D correlations):
          if (leadP.isValid) {
            fillRingObservable(ProxyTypeTag<kLeadP>, leadP, leadPZ, leadPEtaPos, ringCacheLeadP);
            fillIntegratedCuts(ProxyTypeTag<kLeadP>, kinematicLeadPCheck, ringCacheLeadP);
            if (doResamplingPassQA)
              histos.fill(HIST("IntegratedCuts/LeadP/pRingVsResamplingPass"), idxResampling, ringCacheLeadP.ringObservable);
          }

          // Polarization profiles (no proxy needed):
          // 1D TProfiles vs v0phi
          histosRing.fill(HIST("RawPolarization/pPxStarPhi"), v0phiToFillHists, polStarX);
          histosRing.fill(HIST("RawPolarization/pPyStarPhi"), v0phiToFillHists, polStarY);
          histosRing.fill(HIST("RawPolarization/pPzStarPhi"), v0phiToFillHists, polStarZ);
          // 1D TProfiles vs the AEE angle
          histosRing.fill(HIST("RawPolarization/pPxStarPhiAEE"), deltaPhiLambdaProtonStar, polStarX);
          histosRing.fill(HIST("RawPolarization/pPyStarPhiAEE"), deltaPhiLambdaProtonStar, polStarY);
          histosRing.fill(HIST("RawPolarization/pPzStarPhiAEE"), deltaPhiLambdaProtonStar, polStarZ);
          // Lab-frame momentum-plane maps
          histosRing.fill(HIST("RawPolarization/PolMaps/Lab/p2dPxStar_vsPxPy"), v0px, v0py, polStarX);
          histosRing.fill(HIST("RawPolarization/PolMaps/Lab/p2dPyStar_vsPxPy"), v0px, v0py, polStarY);
          histosRing.fill(HIST("RawPolarization/PolMaps/Lab/p2dPzStar_vsPxPy"), v0px, v0py, polStarZ);
          histosRing.fill(HIST("RawPolarization/PolMaps/Lab/h2dCountsVsPxPy"), v0px, v0py);
          histosRing.fill(HIST("RawPolarization/PolMaps/Lab/p2dPxStar_vsPzPx"), v0pz, v0px, polStarX);
          histosRing.fill(HIST("RawPolarization/PolMaps/Lab/p2dPyStar_vsPzPx"), v0pz, v0px, polStarY);
          histosRing.fill(HIST("RawPolarization/PolMaps/Lab/p2dPzStar_vsPzPx"), v0pz, v0px, polStarZ);
          histosRing.fill(HIST("RawPolarization/PolMaps/Lab/h2dCountsVsPzPx"), v0pz, v0px);
          histosRing.fill(HIST("RawPolarization/PolMaps/Lab/p2dPxStar_vsPyPz"), v0py, v0pz, polStarX);
          histosRing.fill(HIST("RawPolarization/PolMaps/Lab/p2dPyStar_vsPyPz"), v0py, v0pz, polStarY);
          histosRing.fill(HIST("RawPolarization/PolMaps/Lab/p2dPzStar_vsPyPz"), v0py, v0pz, polStarZ);
          histosRing.fill(HIST("RawPolarization/PolMaps/Lab/h2dCountsVsPyPz"), v0py, v0pz);
          // Aee frame: acceptance maps (radius = pT^Lambda, azimuth = PhiAEE)
          histosRing.fill(HIST("RawPolarization/PolMaps/Aee/p2dPtStar_vsPxAeePyAee"), v0pxAee, v0pyAee, polStarTAee);
          histosRing.fill(HIST("RawPolarization/PolMaps/Aee/p2dPzStar_vsPxAeePyAee"), v0pxAee, v0pyAee, polStarZ);
          histosRing.fill(HIST("RawPolarization/PolMaps/Aee/h2dCountsVsPxAeePyAee"), v0pxAee, v0pyAee);
          histosRing.fill(HIST("RawPolarization/PolMaps/Aee/p2dPtStar_vsPzPxAee"), v0pz, v0pxAee, polStarTAee);
          histosRing.fill(HIST("RawPolarization/PolMaps/Aee/p2dPzStar_vsPzPxAee"), v0pz, v0pxAee, polStarZ);
          histosRing.fill(HIST("RawPolarization/PolMaps/Aee/h2dCountsVsPzPxAee"), v0pz, v0pxAee);
          histosRing.fill(HIST("RawPolarization/PolMaps/Aee/p2dPtStar_vsPyAeePz"), v0pyAee, v0pz, polStarTAee);
          histosRing.fill(HIST("RawPolarization/PolMaps/Aee/p2dPzStar_vsPyAeePz"), v0pyAee, v0pz, polStarZ);
          histosRing.fill(HIST("RawPolarization/PolMaps/Aee/h2dCountsVsPyAeePz"), v0pyAee, v0pz);
          // PrimeV0 frame: the Lambda production plane (a null test)
          histosRing.fill(HIST("RawPolarization/PolMaps/PrimeV0/p2dPxStarPrimeV0_vsPzPt"), v0pz, v0pt, polStarXPrimeV0);
          histosRing.fill(HIST("RawPolarization/PolMaps/PrimeV0/p2dPyStarPrimeV0_vsPzPt"), v0pz, v0pt, polStarYPrimeV0);
          histosRing.fill(HIST("RawPolarization/PolMaps/PrimeV0/p2dPzStar_vsPzPt"), v0pz, v0pt, polStarZ);
          histosRing.fill(HIST("RawPolarization/PolMaps/PrimeV0/h2dCountsVsPzPt"), v0pz, v0pt);

          if (leadJet.isValid) {
            fillRingObservable(ProxyTypeTag<kLeadJet>, leadJet, leadJetZ, leadJetEtaPos, ringCacheLeadJet);
            fillIntegratedCuts(ProxyTypeTag<kLeadJet>, kinematicLeadJetCheck, ringCacheLeadJet);
            if (doResamplingPassQA)
              histos.fill(HIST("IntegratedCuts/LeadJet/pRingVsResamplingPass"), idxResampling, ringCacheLeadJet.ringObservable);
            histos.fill(HIST("IntegratedCuts/LeadJet/pRingVsNV0s"), nLambdaLikeV0s, ringCacheLeadJet.ringObservable);

            // Binary search using the pre-fetched axes for delta method of error bar estimation:
            const int binPt = mAxisPt->FindBin(v0pt);
            const int binMass = mAxisMass->FindBin(v0LambdaLikeMass);
            const int binDTheta = mAxisDTheta->FindBin(ringCacheLeadJet.deltaTheta);
            deltaTracker.addV0(ringCacheLeadJet.ringObservable, binPt, binMass, binDTheta);

            if (qaSwitches.doFakePolDiagnosticsQA) {
              // Measuring the AEE differentially (azimuthal efficiency effect, which causes different V0 topologies to be enhanced/suppressed)
              // AEE and HEE correlation:
              histos.fill(HIST("FakePolDrivers/CosHEEVsPhiStar/LeadJet/h2dFakePolCounts_CosThetaVsPhiStar"), cosFakePol, phiStar);
              histos.fill(HIST("FakePolDrivers/CosHEEVsPhiStar/LeadJet/p2dFakePolSignal_CosThetaVsPhiStar"), cosFakePol, phiStar, ringCacheLeadJet.ringObservable);

              // AEE and DCA between daughters correlation:
              histos.fill(HIST("FakePolDrivers/PhiStar/DCACorrelations/LeadJet/h2dFakePolCounts_PhiStarVsDCAdau"), phiStar, dcaDau);
              histos.fill(HIST("FakePolDrivers/PhiStar/DCACorrelations/LeadJet/p2dFakePolSignal_PhiStarVsDCAdau"), phiStar, dcaDau, ringCacheLeadJet.ringObservable);
              histos.fill(HIST("FakePolDrivers/PhiStar/DCACorrelations/LeadJet/p3dFakePolSignal_PhiStarVsDCAdauVsProxyEta"), phiStar, dcaDau, leadJet.eta, ringCacheLeadJet.ringObservable);
              histos.fill(HIST("FakePolDrivers/PhiStar/DCACorrelations/LeadJet/p3dFakePolSignal_PhiStarVsDCAdauVsEtaLambda"), phiStar, dcaDau, v0eta, ringCacheLeadJet.ringObservable);

              // DCA dau to PV correlation:
              // For proton-like daughter:
              histos.fill(HIST("FakePolDrivers/PhiStar/DCACorrelations/LeadJet/h2dFakePolCounts_PhiStarVsDCAProLike"), phiStar, protonLikeDCADauToPV);
              histos.fill(HIST("FakePolDrivers/PhiStar/DCACorrelations/LeadJet/p2dFakePolSignal_PhiStarVsDCAProLike"), phiStar, protonLikeDCADauToPV, ringCacheLeadJet.ringObservable);
              histos.fill(HIST("FakePolDrivers/PhiStar/DCACorrelations/LeadJet/p3dFakePolSignal_PhiStarVsDCAProLikeVsProxyEta"), phiStar, protonLikeDCADauToPV, leadJet.eta, ringCacheLeadJet.ringObservable);
              histos.fill(HIST("FakePolDrivers/PhiStar/DCACorrelations/LeadJet/p3dFakePolSignal_PhiStarVsDCAProLikeVsEtaLambda"), phiStar, protonLikeDCADauToPV, v0eta, ringCacheLeadJet.ringObservable);
              // For pion-like daughter:
              histos.fill(HIST("FakePolDrivers/PhiStar/DCACorrelations/LeadJet/h2dFakePolCounts_PhiStarVsDCAPiLike"), phiStar, pionLikeDCADauToPV);
              histos.fill(HIST("FakePolDrivers/PhiStar/DCACorrelations/LeadJet/p2dFakePolSignal_PhiStarVsDCAPiLike"), phiStar, pionLikeDCADauToPV, ringCacheLeadJet.ringObservable);
              histos.fill(HIST("FakePolDrivers/PhiStar/DCACorrelations/LeadJet/p3dFakePolSignal_PhiStarVsDCAPiLikeVsProxyEta"), phiStar, pionLikeDCADauToPV, leadJet.eta, ringCacheLeadJet.ringObservable);
              histos.fill(HIST("FakePolDrivers/PhiStar/DCACorrelations/LeadJet/p3dFakePolSignal_PhiStarVsDCAPiLikeVsEtaLambda"), phiStar, pionLikeDCADauToPV, v0eta, ringCacheLeadJet.ringObservable);

              histos.fill(HIST("FakePolDrivers/ProxyZ/LeadJet/pRingVsProxyZ"), leadJetZ, ringCacheLeadJet.ringObservable);
              histos.fill(HIST("FakePolDrivers/ProxyZ/LeadJet/pRingOverProxyZ_VsProxyEta"), leadJet.eta, ringObservableOverLeadJetZ);
              histos.fill(HIST("FakePolDrivers/ProxyZ/LeadJet/pRingOverProxyZ_VsCosThetaHEE"), cosFakePol, ringObservableOverLeadJetZ);
              histos.fill(HIST("FakePolDrivers/ProxyZ/LeadJet/pRingOverProxyZ_VsPhiStar"), phiStar, ringObservableOverLeadJetZ);
              histos.fill(HIST("FakePolDrivers/ProxyZ/LeadJet/p2dRingOverProxyZ_VsProxyEtaVsCosThetaHEE"), leadJet.eta, cosFakePol, ringObservableOverLeadJetZ);
              histos.fill(HIST("FakePolDrivers/ProxyZ/LeadJet/p2dRingOverProxyZ_VsProxyEtaVsPhiStar"), leadJet.eta, phiStar, ringObservableOverLeadJetZ);

              // HEE resolved in the five mass regions, so a peak-only structure can be told apart from a flat artifact:
              const bool isAEEPhiPos = deltaPhiLambdaProtonStar >= 0;
              histos.fill(HIST("FakePolDrivers/CosHEE/ByMass/h2dCountsCosThetaHEEVsFullMass"), cosFakePol, v0LambdaLikeMass);
              histos.fill(HIST("FakePolDrivers/CosHEE/ByMass/p2dRingCosThetaHEEVsFullMass"), cosFakePol, v0LambdaLikeMass, ringCacheLeadJet.ringObservable);
              histos.fill(HIST("FakePolDrivers/CosHEE/ByMass/p2dRingCosThetaHEEVsMass"), cosFakePol, v0LambdaLikeMass, ringCacheLeadJet.ringObservable);
              if (isAEEPhiPos)
                histos.fill(HIST("FakePolDrivers/CosHEE/ByMass/p2dRingCosThetaHEEVsMassPosPhiAEE"), cosFakePol, v0LambdaLikeMass, ringCacheLeadJet.ringObservable);
              else
                histos.fill(HIST("FakePolDrivers/CosHEE/ByMass/p2dRingCosThetaHEEVsMassNegPhiAEE"), cosFakePol, v0LambdaLikeMass, ringCacheLeadJet.ringObservable);
            } // end doFakePolDiagnosticsQA (leading-jet AEE/HEE)
          }
          if (subJet.isValid) {
            fillRingObservable(ProxyTypeTag<kSubJet>, subJet, 0.f, subJetEtaPos, ringCacheSubJet); // (proxyZ unused: no SubJet RingKernel)
            fillIntegratedCuts(ProxyTypeTag<kSubJet>, kinematicSubJetCheck, ringCacheSubJet);
            if (doResamplingPassQA)
              histos.fill(HIST("IntegratedCuts/SubJet/pRingVsResamplingPass"), idxResampling, ringCacheSubJet.ringObservable);
          }

          if (qaSwitches.doFakePolDiagnosticsQA) {
            // Filling eta dependence QAs of the result (both for V0 and jet proxy):
            if (leadJet.isValid)
              fillFakePolDiagnostics(ProxyTypeTag<kLeadJet>, leadJet, leadJetEtaPos, leadJetEtaStrict, ringCacheLeadJet);
            if (subJet.isValid)
              fillFakePolDiagnostics(ProxyTypeTag<kSubJet>, subJet, subJetEtaPos, subJetEtaStrict, ringCacheSubJet);
            if (leadP.isValid)
              fillFakePolDiagnostics(ProxyTypeTag<kLeadP>, leadP, leadPEtaPos, false, ringCacheLeadP); // (no strict variant for the LeadP)
          } // end doFakePolDiagnosticsQA (eta-dependence block)
        } // end v0s loop

        // Flush the Delta Method accumulators to the actual O2 histograms (once per collision):
        if (deltaTracker.nInt > 0) {
          histosRing.fill(HIST("Ring/DeltaMethod/hIntegrated"), 0.5, deltaTracker.rInt);
          histosRing.fill(HIST("Ring/DeltaMethod/hIntegrated"), 1.5, static_cast<double>(deltaTracker.nInt));
          histosRing.fill(HIST("Ring/DeltaMethod/hIntegrated"), 2.5, deltaTracker.rInt * deltaTracker.rInt);
          histosRing.fill(HIST("Ring/DeltaMethod/hIntegrated"), 3.5, static_cast<double>(deltaTracker.nInt * deltaTracker.nInt));
          histosRing.fill(HIST("Ring/DeltaMethod/hIntegrated"), 4.5, deltaTracker.rInt * deltaTracker.nInt);
        }
        fillDeltaMethodComponents(HIST("Ring/DeltaMethod/h2dLambdaPtVsDeltaComp"), deltaTracker.rPt, deltaTracker.nPt, mAxisPt);
        fillDeltaMethodComponents(HIST("Ring/DeltaMethod/h2dMassVsDeltaComp"), deltaTracker.rMass, deltaTracker.nMass, mAxisMass);
        fillDeltaMethodComponents(HIST("Ring/DeltaMethod/h2dDeltaThetaVsDeltaComp"), deltaTracker.rDtheta, deltaTracker.nDtheta, mAxisDTheta);
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
