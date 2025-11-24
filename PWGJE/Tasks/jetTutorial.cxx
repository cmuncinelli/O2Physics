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

// jet tutorial task for hands on tutorial session (16/11/2024)
//
/// \author Nima Zardoshti <nima.zardoshti@cern.ch>
//

#include "PWGJE/Core/JetDerivedDataUtilities.h"
#include "PWGJE/Core/JetUtilities.h"
#include "PWGJE/DataModel/Jet.h"
#include "PWGJE/DataModel/JetReducedData.h"
#include "PWGJE/DataModel/JetSubtraction.h"

#include "Common/Core/RecoDecay.h"
#include "Common/DataModel/TrackSelectionTables.h"

#include "Framework/ASoA.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisTask.h"
#include "Framework/HistogramRegistry.h"
#include <CommonConstants/MathConstants.h>
#include <Framework/Configurable.h>
#include <Framework/HistogramSpec.h>
#include <Framework/InitContext.h>
#include <Framework/runDataProcessing.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;

struct JetTutorialTask {
  HistogramRegistry registry{"registry",
                             {{"h_collisions", "event status;event status;entries", {HistType::kTH1F, {{4, 0.0, 4.0}}}},
                              {"h_track_pt", "track pT;#it{p}_{T,track} (GeV/#it{c});entries", {HistType::kTH1F, {{200, 0., 200.}}}},
                              {"h_track_eta", "track #eta;#eta_{track};entries", {HistType::kTH1F, {{100, -1.0, 1.0}}}},
                              {"h_track_phi", "track #varphi;#varphi_{track};entries", {HistType::kTH1F, {{80, -1.0, 7.}}}},
                              {"h_track_chi2PerCluster", "track #chi^{2} per cluste ;#chi^{2};entries", {HistType::kTH1F, {{100, 0, 40}}}},
                              {"h_jet_pt", "jet pT;#it{p}_{T,jet} (GeV/#it{c});entries", {HistType::kTH1F, {{200, 0., 200.}}}},
                              {"h_jet_eta", "jet #eta;#eta_{jet};entries", {HistType::kTH1F, {{100, -1.0, 1.0}}}},
                              {"h_jet_phi", "jet #phi;#phi_{jet};entries", {HistType::kTH1F, {{80, -1.0, 7.}}}},
                              {"h_jet_pt_rhosub", "jet pT bkg sub;#it{p}_{T,jet} (GeV/#it{c});entries", {HistType::kTH1F, {{200, 0., 200.}}}},
                              {"h_jet_pt_constsub", "jet pT bkg sub;#it{p}_{T,jet} (GeV/#it{c});entries", {HistType::kTH1F, {{200, 0., 200.}}}},
                              {"h_part_jet_pt", "particle level jet pT;#it{p}_{T,jet part} (GeV/#it{c});entries", {HistType::kTH1F, {{200, 0., 200.}}}},
                              {"h_part_jet_eta", "particle level jet #eta;#eta_{jet part};entries", {HistType::kTH1F, {{100, -1.0, 1.0}}}},
                              {"h_part_jet_phi", "particle level jet #phi;#phi_{jet part};entries", {HistType::kTH1F, {{80, -1.0, 7.}}}},
                              {"h_jet_ntracks", "jet N tracks;N_{jet tracks};entries", {HistType::kTH1F, {{40, -0.5, 39.5}}}},
                              {"h_jet_angularity", "jet angularity ;#lambda_{1};entries", {HistType::kTH1F, {{5, 0.0, 0.5}}}},
                              {"h_jet_angularity_constsub", "jet angularity bkg sub;#lambda_{1};entries", {HistType::kTH1F, {{5, 0.0, 0.5}}}},
                              {"h_full_jet_pt", "jet pT;#it{p}_{T,jet} (GeV/#it{c});entries", {HistType::kTH1F, {{200, 0., 200.}}}},
                              {"h_full_jet_eta", "jet #eta;#eta_{jet};entries", {HistType::kTH1F, {{100, -1.0, 1.0}}}},
                              {"h_full_jet_phi", "jet #phi;#phi_{jet};entries", {HistType::kTH1F, {{80, -1.0, 7.}}}},
                              {"h_full_jet_ntracks", "jet N tracks;N_{jet tracks};entries", {HistType::kTH1F, {{40, -0.5, 39.5}}}},
                              {"h_full_jet_nclusters", "jet N clusters;N_{jet clusters};entries", {HistType::kTH1F, {{40, -0.5, 39.5}}}},
                              {"h_full_jet_angularity", "jet angularity ;#lambda_{1};entries", {HistType::kTH1F, {{5, 0.0, 0.5}}}},
                              {"h_part_jet_angularity", "jet angularity ;#lambda_{1};entries", {HistType::kTH1F, {{5, 0.0, 0.5}}}},
                              {"h_recoil_jet_pt", "jet pT;#it{p}_{T,jet} (GeV/#it{c});entries", {HistType::kTH1F, {{200, 0., 200.}}}},
                              {"h_recoil_jet_eta", "jet #eta;#eta_{jet};entries", {HistType::kTH1F, {{100, -1.0, 1.0}}}},
                              {"h_recoil_jet_phi", "jet #phi;#phi_{jet};entries", {HistType::kTH1F, {{80, -1.0, 7.}}}},
                              {"h_recoil_jet_dphi", "hadron-jet #Delta#phi;#Delta#phi_{jet,trigger hadron};entries", {HistType::kTH1F, {{40, -2.0, 2.0}}}},
                              {"h_matched_jets_pt", "#it{p}_{T,jet part}; #it{p}_{T,jet det}", {HistType::kTH2F, {{100, 0., 20.}, {100, 0., 20.0}}}},
                              {"h_matched_jets_eta", "#eta_{jet part}; #eta_{jet det}", {HistType::kTH2F, {{100, -1.0, 1.0}, {100, -1.0, 1.0}}}},
                              {"h_matched_jets_phi", "#phi_{jet part}; #phi_{jet det}", {HistType::kTH2F, {{80, -1.0, 7.}, {80, -1.0, 7.}}}}}};

  Configurable<float> vertexZCut{"vertexZCut", 10.0f, "Accepted z-vertex range"};

  Configurable<float> jetPtMin{"jetPtMin", 5.0, "minimum jet pT cut"};
  Configurable<float> jetR{"jetR", 0.4, "jet resolution parameter"};

  Configurable<std::string> eventSelections{"eventSelections", "sel8", "choose event selection"}; // Could also be "sel8+sel7" or any combo like that, with "+" as spacings
  Configurable<std::string> trackSelections{"trackSelections", "globalTracks", "set track selections"};

  Configurable<float> kappa{"kappa", 1.0, "angularity kappa"};
  Configurable<float> alpha{"alpha", 1.0, "angularity alpha"};

  Configurable<std::string> triggerMasks{"triggerMasks", "", "possible JE Trigger masks: fJetChLowPt,fJetChHighPt,fTrackLowPt,fTrackHighPt,fJetD0ChLowPt,fJetD0ChHighPt,fJetLcChLowPt,fJetLcChHighPt,fEMCALReadout,fJetFullHighPt,fJetFullLowPt,fJetNeutralHighPt,fJetNeutralLowPt,fGammaVeryHighPtEMCAL,fGammaVeryHighPtDCAL,fGammaHighPtEMCAL,fGammaHighPtDCAL,fGammaLowPtEMCAL,fGammaLowPtDCAL,fGammaVeryLowPtEMCAL,fGammaVeryLowPtDCAL"};

  std::vector<int> eventSelectionBits;
  int trackSelection = -1;
  std::vector<int> triggerMaskBits;

  void init(o2::framework::InitContext&)
  {
    eventSelectionBits = jetderiveddatautilities::initialiseEventSelectionBits(static_cast<std::string>(eventSelections)); // Fed with the eventSelections string!
    trackSelection = jetderiveddatautilities::initialiseTrackSelection(static_cast<std::string>(trackSelections));
    triggerMaskBits = jetderiveddatautilities::initialiseTriggerMaskBits(triggerMasks);
  }

  Filter jetCuts = aod::jet::pt > jetPtMin&& aod::jet::r == nround(jetR.node() * 100.0f); // This applies a cut to the jet properties.
                                                                                          // Notice you have to select a particular jet R for your selections,
                                                                                          // because the way their code runs is it stores smaller radii jets
                                                                                          // in the same big table, instead of just a single R for each dataset.
                                                                                          // We also save the jet radius as a multiple of 100 of the actual value
                                                                                          // of the radius, because that helps avoid floating round errors!
                                                                                          // We always store it as 100 times bigger than it actually is!
                                                                                          // So we pick a jetR as 0.4, and they store it as 40 inside their codes,
                                                                                          // so you need to multiply it by 100 !
  Filter collisionFilter = nabs(aod::jcollision::posZ) < vertexZCut;
  Filter mcCollisionFilter = nabs(aod::jmccollision::posZ) < vertexZCut;

  Preslice<soa::Filtered<aod::ChargedMCParticleLevelJets>> perMcCollisionJets = aod::jet::mcCollisionId; // Gives you the jets that come only from this
                                                                                                         // particular MC collision!

  void processCollisions(aod::JetCollision const& collision, aod::JetTracks const& tracks) // Notice the use of JetCollision, not JetCollisions. Check the Jet.h file to see the specific on what tables this subscribes to
    // "Collision" will just give us the tracks of this particular collision. "Collisions" will give us the tracks for ALL collisions
  {

    registry.fill(HIST("h_collisions"), 0.5);
    if (!jetderiveddatautilities::selectCollision(collision, eventSelectionBits)) { // Already fed with the event selection string from before!
      return; // Just skips this particular collision, if useless
    }
    registry.fill(HIST("h_collisions"), 1.5);
    for (auto const& track : tracks) {
      if (!jetderiveddatautilities::selectTrack(track, trackSelection)) { // Same kind of selection as for the events, but now for tracks. Uses this "selectTrack" function
        continue;
      }
      registry.fill(HIST("h_track_pt"), track.pt());
      registry.fill(HIST("h_track_eta"), track.eta());
      registry.fill(HIST("h_track_phi"), track.phi());
    }
  }
  PROCESS_SWITCH(JetTutorialTask, processCollisions, "process JE collisions", false);

  // This is an example on how to use external tracks, from other tables.
  // You join with the JTrackPIs ("PI" = "Parent Index"), in such a way that you can join with other original track tables.
  // We are also applying the filters previously defined, such as the "collisionFilter" from before.
  void processCollisionsWithExternalTracks(soa::Filtered<aod::JetCollisions>::iterator const& collision, soa::Join<aod::JetTracks, aod::JTrackPIs> const& tracks, soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksDCA, aod::TrackSelection> const&)
  // To run this, activate "processCollisionsWithExternalTracks" inside the dpl-config-base.json and deactivate the "processCollisions" part
  {
    // Filter could also be if (std::abs(collision.posz()) < 10) and the such, but the Filtered process is much faster

    registry.fill(HIST("h_collisions"), 0.5);
    // This is one utility function defined for JetCollisions. For this simple example, you could just use the original data tables and use the AO2D.
    // The only use case you would actually need something like this is when running on jet derived data. Those derived data need these kinds of
    // tracks and collisions. For more complicated use cases, where there is something more related to jets, you would need the subscription to these
    // jet-only tables, collisions and tracks.
    // This is a commitment when using the Jet Framework, and they do this to have the smallest possible AO2Ds
    if (!jetderiveddatautilities::selectCollision(collision, eventSelectionBits)) {
      return;
    }
    registry.fill(HIST("h_collisions"), 1.5);
    for (auto const& track : tracks) { // Here tracks is JTracks though
      if (!jetderiveddatautilities::selectTrack(track, trackSelection)) { // This is one utility function defined for JetTracks
        continue;
      }
      registry.fill(HIST("h_track_pt"), track.pt());
      registry.fill(HIST("h_track_eta"), track.eta());
      registry.fill(HIST("h_track_phi"), track.phi());
        // This is how you access the original track from outside of the Jets framework, and this can give you the needed information
          // This CANNOT be ran on the derived data, so only original AO2Ds which are way bigger and slower to run jets on,
          // but for most use cases you will probably not need to write 
      auto originalTrack = track.track_as<soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksDCA, aod::TrackSelection>>();
      registry.fill(HIST("h_track_chi2PerCluster"), originalTrack.tpcChi2NCl());
    }
  }
  PROCESS_SWITCH(JetTutorialTask, processCollisionsWithExternalTracks, "process JE collisions with access to the original track table", false);

  void processDataCharged(soa::Filtered<aod::JetCollisions>::iterator const& collision, soa::Filtered<aod::ChargedJets> const& jets)
  {
    if (!jetderiveddatautilities::selectCollision(collision, eventSelectionBits)) { // Already done for the derived data, but useful for full AO2Ds
      return;
    }
    for (auto& jet : jets) { // Runs for all charged jets only!
      registry.fill(HIST("h_jet_pt"), jet.pt());
      registry.fill(HIST("h_jet_eta"), jet.eta());
      registry.fill(HIST("h_jet_phi"), jet.phi());
    }
  }
  PROCESS_SWITCH(JetTutorialTask, processDataCharged, "charged jets in data", false);

  void processMCDetectorLevelCharged(soa::Filtered<aod::JetCollisions>::iterator const& collision, soa::Filtered<aod::ChargedMCDetectorLevelJets> const& jets)
  {
    if (!jetderiveddatautilities::selectCollision(collision, eventSelectionBits)) { // Here you might use "sel8MC" instead of "sel8", because it has a different set of filters
      return;
    }
    for (auto& jet : jets) {
      registry.fill(HIST("h_jet_pt"), jet.pt());
      registry.fill(HIST("h_jet_eta"), jet.eta());
      registry.fill(HIST("h_jet_phi"), jet.phi());
    }
  }
  PROCESS_SWITCH(JetTutorialTask, processMCDetectorLevelCharged, "charged jets in detector level MC", false);

  // We need to attribute a weighting scheme for the MC detector-level data
  // JetCollisionsMCD is the same as the previous JetCollisions, but also with the table of JMcCollisionsLbs table joined along it.
  // Precisely: "using JetCollisionsMCD = o2::soa::Join<JetCollisions, JMcCollisionLbs>;"
  void processMCDetectorLevelWeightedCharged(soa::Filtered<aod::JetCollisionsMCD>::iterator const& collision, aod::JetMcCollisions const&, soa::Filtered<aod::ChargedMCDetectorLevelJets> const& jets)
  {
    if (!jetderiveddatautilities::selectCollision(collision, eventSelectionBits)) {
      return;
    }
    for (auto& jet : jets) {
      registry.fill(HIST("h_jet_pt"), jet.pt(), collision.mcCollision().weight()); // The .mcCollision() part links to the particle-level collision, not the just detector-level simulation
                                                                                   // This weight also lives inside JetMcCollisions, so that's why we subscribed to it.
      registry.fill(HIST("h_jet_eta"), jet.eta(), collision.mcCollision().weight());
      registry.fill(HIST("h_jet_phi"), jet.phi(), collision.mcCollision().weight());
    }
  }
  PROCESS_SWITCH(JetTutorialTask, processMCDetectorLevelWeightedCharged, "charged jets in weighted detector level MC", false);


  // Now we use the JetMcCollisions table, which has particle-level (generator-level) collisions and tracks, not detector-level simulations.
  void processMCParticleLevelCharged(soa::Filtered<aod::JetMcCollisions>::iterator const& mcCollision, soa::Filtered<aod::ChargedMCParticleLevelJets> const& jets)
  {
    // Notice there is no "sel8"-like selection at generator level!
    for (auto& jet : jets) {
      registry.fill(HIST("h_part_jet_pt"), jet.pt(), mcCollision.weight()); // What are these weights?
      registry.fill(HIST("h_part_jet_eta"), jet.eta(), mcCollision.weight());
      registry.fill(HIST("h_part_jet_phi"), jet.phi(), mcCollision.weight());
    }
  }
  PROCESS_SWITCH(JetTutorialTask, processMCParticleLevelCharged, "charged jets in particle level MC", false);

  void processMCCharged(soa::Filtered<aod::JetCollisionsMCD>::iterator const& collision, aod::JetMcCollisions const&, soa::Filtered<aod::ChargedMCDetectorLevelJets> const& mcdjets, soa::Filtered<aod::ChargedMCParticleLevelJets> const& mcpjets)
  {
    if (!jetderiveddatautilities::selectCollision(collision, eventSelectionBits)) {
      return;
    }
    for (auto& mcdjet : mcdjets) {
      registry.fill(HIST("h_jet_pt"), mcdjet.pt(), collision.mcCollision().weight());
      registry.fill(HIST("h_jet_eta"), mcdjet.eta(), collision.mcCollision().weight());
      registry.fill(HIST("h_jet_phi"), mcdjet.phi(), collision.mcCollision().weight());
    }
    auto mcpjetsPerCollision = mcpjets.sliceBy(perMcCollisionJets, collision.mcCollisionId()); // Defined in the perMcCollisionJets preslice
      // Matches the particle-level jets to the collision-level jets.
      // Detector-level are different tables and collisions, so you need to join their indices appropriately
      // By the way, grouping when using "Collision" or "Collisions" already does sliceBy internally! So here we are just implementing that manually
    for (auto& mcpjet : mcpjetsPerCollision) {
      registry.fill(HIST("h_part_jet_pt"), mcpjet.pt(), collision.mcCollision().weight());
      registry.fill(HIST("h_part_jet_eta"), mcpjet.eta(), collision.mcCollision().weight());
      registry.fill(HIST("h_part_jet_phi"), mcpjet.phi(), collision.mcCollision().weight());
    }
  }
  PROCESS_SWITCH(JetTutorialTask, processMCCharged, "charged jets in detector and particle level MC", false);

  using JetMCPTable = soa::Filtered<soa::Join<aod::ChargedMCParticleLevelJets, aod::ChargedMCParticleLevelJetConstituents, aod::ChargedMCParticleLevelJetsMatchedToChargedMCDetectorLevelJets>>;
    // Above, a filtered table to match indices
  void processMCMatchedCharged(soa::Filtered<aod::JetCollisionsMCD>::iterator const& collision,
                               soa::Filtered<soa::Join<aod::ChargedMCDetectorLevelJets, aod::ChargedMCDetectorLevelJetConstituents, aod::ChargedMCDetectorLevelJetsMatchedToChargedMCParticleLevelJets>> const& mcdjets,
                               JetMCPTable const&,
                               aod::JetTracks const&, // Detector-level particles
                               aod::JetParticles const&) // Generator-level particles
  // Works on particle-level and detector-level collisions.
  // Then takes detector-level jets.
  // Joins with particle-level tracks, jets and constituents, for *CHARGED* jets.
  // This is performed on the o2-analysis-je-jet-matching-mc-ch matching task, that can match these jets
  // via geometry, via pT matching fractions and the such.
  {
    if (!jetderiveddatautilities::selectCollision(collision, eventSelectionBits)) {
      return;
    }
    for (const auto& mcdjet : mcdjets) {
      for (auto& mcpjet : mcdjet.template matchedJetGeo_as<JetMCPTable>()) { // Usually there is just one particle-level jet matched to a detector-level jet,
                                                                             // but that may not be the case! The matching might not be perfect!
        // You might need a pT + geometrical matching when doing more complicated analyses such as PbPb collision checks.
        registry.fill(HIST("h_matched_jets_pt"), mcpjet.pt(), mcdjet.pt(), collision.mcCollision().weight());
          // Compares the detector-level and generator-level jet pT that were matched via geometry, and gives it a weight based on the mcCollision.
        registry.fill(HIST("h_matched_jets_pt"), mcpjet.phi(), mcdjet.phi(), collision.mcCollision().weight());
        registry.fill(HIST("h_matched_jets_pt"), mcpjet.eta(), mcdjet.eta(), collision.mcCollision().weight());
      }
    }
  }
  PROCESS_SWITCH(JetTutorialTask, processMCMatchedCharged, "matched detector and particle level charged jets", false);

  // Now a process that can help us see the jet constituents from a Charged Jet jet-finder output (ChargedJetConstituents)
  void processDataSubstructureCharged(soa::Filtered<aod::JetCollisions>::iterator const& collision, soa::Filtered<soa::Join<aod::ChargedJets, aod::ChargedJetConstituents>> const& jets, aod::JetTracks const&)
  // For heavy-flavour analyses, you could also include aod::CandidatesD0Data const& for the analysis, running with the correspondent HF jet-finder to produce these useful tables.
  {
    if (!jetderiveddatautilities::selectCollision(collision, eventSelectionBits)) {
      return;
    }
    for (auto& jet : jets) {
      registry.fill(HIST("h_jet_pt"), jet.pt());
      registry.fill(HIST("h_jet_eta"), jet.eta());
      registry.fill(HIST("h_jet_phi"), jet.phi());
      registry.fill(HIST("h_jet_ntracks"), jet.tracksIds().size());
      double angularity = 0.0; // We are going to try and measure the angularity of the substructures of the jet.
      for (auto& jetConstituent : jet.tracks_as<aod::JetTracks>()) { // This is the way to get jet constituent objects!
        angularity += std::pow(jetConstituent.pt(), kappa) * std::pow(jetutilities::deltaR(jet, jetConstituent), alpha);
      }
      angularity /= (jet.pt() * (jet.r() / 100.f));
      registry.fill(HIST("h_jet_angularity"), angularity);
    }
  }
  PROCESS_SWITCH(JetTutorialTask, processDataSubstructureCharged, "charged jet substructure", false);

  // Processes full jets! Look at the "jet-finder-mcp-full" and MC-equivalents for the full specs.
  // Now, just be careful that the variables of hadronic selection and the such start mattering a lot more!
  // A new interesting variable to look at is "doEMCALEventSelectionChargedJets" or "doEMCALEventSelection" to look at
  // photon jets and photon-jet events using only events where EMCal was on!
  void processDataFull(soa::Filtered<aod::JetCollisions>::iterator const&, soa::Filtered<aod::FullJets> const& jets)
  {
    for (auto& jet : jets) {
      registry.fill(HIST("h_jet_pt"), jet.pt());
      registry.fill(HIST("h_jet_eta"), jet.eta());
      registry.fill(HIST("h_jet_phi"), jet.phi());
    }
  }
  PROCESS_SWITCH(JetTutorialTask, processDataFull, "full jets in data", false);

  void processDataSubstructureFull(soa::Filtered<aod::JetCollisions>::iterator const&, soa::Filtered<soa::Join<aod::FullJets, aod::FullJetConstituents>> const& jets, aod::JetTracks const&, aod::JetClusters const&)
  {
    for (auto& jet : jets) {
      registry.fill(HIST("h_full_jet_pt"), jet.pt());
      registry.fill(HIST("h_full_jet_eta"), jet.eta());
      registry.fill(HIST("h_full_jet_phi"), jet.phi());
      registry.fill(HIST("h_full_jet_ntracks"), jet.tracksIds().size());
      registry.fill(HIST("h_full_jet_nclusters"), jet.clustersIds().size());
      double angularity = 0.0;
      for (auto& jetConstituent : jet.tracks_as<aod::JetTracks>()) {
        angularity += std::pow(jetConstituent.pt(), kappa) * std::pow(jetutilities::deltaR(jet, jetConstituent), alpha);
      }

      for (auto& jetCluster : jet.clusters_as<aod::JetClusters>()) {
        angularity += std::pow(jetCluster.energy(), kappa) * std::pow(jetutilities::deltaR(jet, jetCluster), alpha);
      }

      registry.fill(HIST("h_full_jet_angularity"), angularity / (jet.pt() * round(jet.r() * 100.0f)));
    }
  }
  PROCESS_SWITCH(JetTutorialTask, processDataSubstructureFull, "full jet substructure", false);

  void processMCParticleLevelSubstructureFull(soa::Filtered<aod::JetMcCollisions>::iterator const& mcCollision, soa::Filtered<soa::Join<aod::FullMCParticleLevelJets, aod::FullMCParticleLevelJetConstituents>> const& jets, aod::JetParticles const&)
  {
    for (auto& jet : jets) {
      double angularity = 0.0;
      for (auto& jetConstituent : jet.tracks_as<aod::JetParticles>()) {
        angularity += std::pow(jetConstituent.pt(), kappa) * std::pow(jetutilities::deltaR(jet, jetConstituent), alpha);
      }
      angularity /= (jet.pt() * (jet.r() / 100.f));
      registry.fill(HIST("h_part_jet_angularity"), angularity, mcCollision.weight());
    }
  }
  PROCESS_SWITCH(JetTutorialTask, processMCParticleLevelSubstructureFull, "full particle level jet substructure", false);

  // Useful equation to study photon-jets and jet-hadron correlations!
  void processRecoilDataCharged(soa::Filtered<aod::JetCollisions>::iterator const& collision, soa::Filtered<aod::ChargedJets> const& jets, aod::JetTracks const& tracks)
  {
    if (!jetderiveddatautilities::selectCollision(collision, eventSelectionBits)) {
      return;
    }
    bool selectedEvent = false;
    double leadingTrackpT = 0.0;
    double leadingTrackPhi = 0.0;
    for (auto& track : tracks) {
      if (track.pt() > 6.0 && track.pt() < 10.0) { // An interesting loop to select leading particles in jets.
                                                   // Should look at this in PbPb examples or tasks!
        if (track.pt() > leadingTrackpT) {
          leadingTrackpT = track.pt();
          leadingTrackPhi = track.phi();
          selectedEvent = true;
        }
      }
    }
    if (!selectedEvent) {
      return;
    }
    for (auto& jet : jets) {
      // if (std::abs(RecoDecay::constrainAngle(jet.phi() - leadingTrackPhi, -o2::constants::math::PIHalf)) > 0.6) {
      // Should actually be PI, not PIHalf, right? After all, we are measuring jet recoil
      if (std::abs(RecoDecay::constrainAngle(jet.phi() - leadingTrackPhi, -o2::constants::math::PI)) > 0.6) {
        registry.fill(HIST("h_recoil_jet_pt"), jet.pt()); // Measures jet recoil!
        registry.fill(HIST("h_recoil_jet_eta"), jet.eta());
        registry.fill(HIST("h_recoil_jet_phi"), jet.phi());
        registry.fill(HIST("h_recoil_jet_dphi"), jet.phi() - leadingTrackPhi);
      }
    }
  }
  PROCESS_SWITCH(JetTutorialTask, processRecoilDataCharged, "hadron-recoil charged jets", false);

  // Now a background subtraction analysis
  void processDataRhoAreaSubtractedCharged(soa::Filtered<soa::Join<aod::JetCollisions, aod::BkgChargedRhos>>::iterator const& collision, soa::Filtered<aod::ChargedJets> const& jets)
  {
    if (!jetderiveddatautilities::selectCollision(collision, eventSelectionBits)) {
      return;
    }
    for (auto jet : jets) {
      registry.fill(HIST("h_jet_pt"), jet.pt());
      registry.fill(HIST("h_jet_pt_rhosub"), jet.pt() - (collision.rho() * jet.area())); // The simplest area-based background subtraction in jet pT!
                                                                                         // In Run3, you can only do background subtraction on the event itself (as it should)!
      // There is also a perpendicular current background in double = jetutilities::estimateRhoPerpCone(tracks, jets, jet.r()/100.), which will give
      // you a background subtraction for a selected jet, tracks and jet radius
      // (or use that from jetutilities::estimateRhoPerpendicular)
      // To use this task, you would need something like o2-analysis-je-rho or something, and a "subtractor-event-wise" subtractor in your workflow.
      // This last subtractor task will produced subtracted jet tables! (JetTracksSub, a table linked to the collision and giving subtracted tracks)
        // Just notice that in that subtable you can't do track selection anymore, as track subtraction makes you lose the identity of each track.
        // You also need to turn "processChargedEvtWiseSubJets" on in the dpl-config-base.json
      registry.fill(HIST("h_jet_eta"), jet.eta());
      registry.fill(HIST("h_jet_phi"), jet.phi());
    }
  }
  PROCESS_SWITCH(JetTutorialTask, processDataRhoAreaSubtractedCharged, "charged rho-area  subtracted jets", false);

  void processDataConstituentSubtractedCharged(soa::Filtered<aod::JetCollisions>::iterator const& collision, soa::Filtered<aod::ChargedEventWiseSubtractedJets> const& jets)
  {
    if (!jetderiveddatautilities::selectCollision(collision, eventSelectionBits)) {
      return;
    }
    for (auto jet : jets) {
      registry.fill(HIST("h_jet_pt_constsub"), jet.pt());
      registry.fill(HIST("h_jet_eta"), jet.eta());
      registry.fill(HIST("h_jet_phi"), jet.phi());
    }
  }
  PROCESS_SWITCH(JetTutorialTask, processDataConstituentSubtractedCharged, "charged constituent subtracted jets", false);

  void processDataConstituentSubtractedSubstructureCharged(soa::Filtered<aod::JetCollisions>::iterator const& collision, soa::Filtered<soa::Join<aod::ChargedEventWiseSubtractedJets, aod::ChargedEventWiseSubtractedJetConstituents>> const& jets, aod::JetTracksSub const&)
  {
    if (!jetderiveddatautilities::selectCollision(collision, eventSelectionBits)) {
      return;
    }
    for (auto jet : jets) {
      registry.fill(HIST("h_jet_pt_constsub"), jet.pt());
      registry.fill(HIST("h_jet_eta"), jet.eta());
      registry.fill(HIST("h_jet_phi"), jet.phi());
      registry.fill(HIST("h_jet_ntracks"), jet.tracksIds().size());
      double angularity = 0.0;
      for (auto& jetConstituent : jet.tracks_as<aod::JetTracksSub>()) {
        angularity += std::pow(jetConstituent.pt(), kappa) * std::pow(jetutilities::deltaR(jet, jetConstituent), alpha);
      }
      angularity /= (jet.pt() * (jet.r() / 100.f));
      registry.fill(HIST("h_jet_angularity_constsub"), angularity);
    }
  }
  PROCESS_SWITCH(JetTutorialTask, processDataConstituentSubtractedSubstructureCharged, "charged constituent subtracted jet substructure", false);

  void processDataTriggered(soa::Filtered<aod::JetCollisions>::iterator const& collision, soa::Filtered<aod::ChargedJets> const& jets)
  {
    if (!jetderiveddatautilities::selectCollision(collision, eventSelectionBits) || !jetderiveddatautilities::selectTrigger(collision, triggerMaskBits)) {
      // triggerMaskBits is defined in the same way as in event selections and the such.
      // Look at the triggerMasks vector to see what kinds of triggered events you can use, and do notice you can match many triggers at the same time.
      // For triggered samples, should also use the "includeTriggers" flag in the dpl-config.json, which will be needed for full data runs (full AO2Ds, not derived data).
      // By activating this, you will skip the events that don't pass the trigger selections you are interested in and will save resources.
      return;
    }
    for (auto& jet : jets) {
      registry.fill(HIST("h_jet_pt"), jet.pt());
      registry.fill(HIST("h_jet_eta"), jet.eta());
      registry.fill(HIST("h_jet_phi"), jet.phi());
    }
  }
  PROCESS_SWITCH(JetTutorialTask, processDataTriggered, "jets triggered", false);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc) { return WorkflowSpec{adaptAnalysisTask<JetTutorialTask>(cfgc, TaskName{"jet-tutorial"})}; }
