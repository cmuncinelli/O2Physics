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
/// \author Nima Zardoshti <nima.zardoshti@cern.ch>, CERN
// WARNING! This code does not run with pipelining in the *-medium.root derived data provided by David, but it does work on the small and large ones, for some reason...

// O2 includes
#include "ReconstructionDataFormats/Track.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h" // For the filter to work
#include "Common/DataModel/TrackSelectionTables.h"
#include "DataModel/DerivedExampleTable.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions; // Filter namespace

#include "Framework/runDataProcessing.h"

struct DerivedBasicConsumer {
  /// Function to aid in calculating delta-phi
  /// \param phi1 first phi value
  /// \param phi2 second phi value
  Double_t ComputeDeltaPhi(Double_t phi1, Double_t phi2)
  {
    Double_t deltaPhi = phi1 - phi2;
    if (deltaPhi < -TMath::Pi() / 2.) {
      deltaPhi += 2. * TMath::Pi();
    }
    if (deltaPhi > 3 * TMath::Pi() / 2.) {
      deltaPhi -= 2. * TMath::Pi();
    }
    return deltaPhi;
  }

    // Defining a configurable axis for easier manipulation later on:
  ConfigurableAxis axisPtQA{"axisPtQA", {VARIABLE_WIDTH, 0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f,
  0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f, 2.2f,
  2.4f, 2.6f, 2.8f, 3.0f, 3.2f, 3.4f, 3.6f, 3.8f, 4.0f, 4.4f, 4.8f, 5.2f, 5.6f, 6.0f, 6.5f, 7.0f,
  7.5f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 17.0f, 19.0f, 21.0f, 23.0f, 25.0f,
  30.0f, 35.0f, 40.0f, 50.0f}, "pt axis for QA histograms"};

  // Defining a quality filter to be applied in all of my data:
  Filter collZfilter = nabs(aod::collision::posZ) < 10.0f;

  // Defining a partition in my data's pT:
    // The associated tracks are the range at which we will check correlation, and the trigger tracks are the reference for the jet positions (specially, the near-side jet position)
  SliceCache cache;

  Partition<aod::DrTracks> associatedTracks = aod::exampleTrackSpace::pt < 6.0f && aod::exampleTrackSpace::pt > 4.0f; // exampleTrackSpace is the DataModel for this particular exercise
  Partition<aod::DrTracks> triggerTracks = aod::exampleTrackSpace::pt > 6.0f;

  // Histogram registry: an object to hold your histograms
  HistogramRegistry histos{"histos", {}, OutputObjHandlingPolicy::AnalysisObject};

  void init(InitContext const&)
  {
    // define axes you want to use
    const AxisSpec axisCounter{1, 0, +1, ""};
    // const AxisSpec axisPtQA{500, 0, 10, "p_{T}"};
    const AxisSpec axisDeltaPhi{100, -0.5*TMath::Pi(), +1.5*TMath::Pi(), "#Delta#phi"};
      // Including a 2D histogram for the correlations!
    const AxisSpec axisDeltaEta{100, -1.0, +1.0, "#Delta#eta"};

    histos.add("eventCounter", "eventCounter", kTH1F, {axisCounter});
    histos.add("ptAssoHistogram", "ptAssoHistogram", kTH1F, {axisPtQA});
    histos.add("ptTrigHistogram", "ptTrigHistogram", kTH1F, {axisPtQA});
    histos.add("correlationFunction", "correlationFunction", kTH1F, {axisDeltaPhi});
    histos.add("correlationFunction2d", "correlationFunction2d", kTH2F, {axisDeltaPhi, axisDeltaEta});
  }

  // void process(aod::DrCollision const& /*collision*/)
  void process(soa::Filtered<aod::DrCollisions>::iterator const& collision, aod::DrTracks const& tracks) // Iterating on all collisions that passed the given quality check. Also subscribed to all of their tracks, which will be selected later on!
  {
    histos.fill(HIST("eventCounter"), 0.5);

    // Actually grouping the previous partitions in a per-collision level loop (we need to have a different \phi_0 (jet_pos) for each collision, thus the need to group the tables by collision to loop on them)
      // Do notice, also, that the partitioning rules are stored in the SliceCache object "cache"!
    auto assoTracksThisCollision = associatedTracks->sliceByCached(aod::exampleTrackSpace::drCollisionId, collision.globalIndex(), cache);
    auto trigTracksThisCollision = triggerTracks->sliceByCached(aod::exampleTrackSpace::drCollisionId, collision.globalIndex(), cache);

    // Filling two QA histograms for this particular selection:
    for (auto& track : assoTracksThisCollision) histos.fill(HIST("ptAssoHistogram"), track.pt());
    for (auto& track : trigTracksThisCollision) histos.fill(HIST("ptTrigHistogram"), track.pt());

      // Now using structured binding (some new stuff from C++17) to select every possible (i, j) pair (with i obviously != j, because the space in which i lives is different from the one j lives on: trigger vs associated)
        // In principle, you can use this loop to go through all particles and avoid the loops above, but that would be stupid!
        // This one repeats i particles for every possible combination with different j particles, so you would have to normalize each particle's weight to unit, i.e., divide all the contributions of
        // i by the number of j's for which it was repeated. The computational cost of adding many times the same particle to a histogram, then normalizing it, is much greater than just reading through
        // the whole table one more time! Thus, I will keep two separate loops (or, to be more precise, three loops: two for the QA-type loop, and this one for the combination-type loop).
    for (auto& [trigger, associated] : combinations(o2::soa::CombinationsFullIndexPolicy(trigTracksThisCollision, assoTracksThisCollision))){
      histos.fill(HIST("correlationFunction"), ComputeDeltaPhi(trigger.phi(),associated.phi()));
      histos.fill(HIST("correlationFunction2d"), ComputeDeltaPhi(trigger.phi(),associated.phi()), trigger.eta() - associated.eta());
      // In the case of eta, as it is not periodic, you can just go ahead and use the subtraction of etas itself.
      // No need to make sure it falls within a specific, periodic, interval.
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec workflow{adaptAnalysisTask<DerivedBasicConsumer>(cfgc, TaskName{"derived-basic-consumer"})};
  return workflow;
}
