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
///
/// \brief This task is an empty skeleton that fills a simple eta histogram.
///        it is meant to be a blank page for further developments.
/// \author everyone

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Common/DataModel/TrackSelectionTables.h" // New header needed for the dcaXY information!

using namespace o2;
using namespace o2::framework;

struct myExampleTask2 { // There should actually be no need to rename the struct, but renamed for further reference.
  // Histogram registry: an object to hold your histograms
  HistogramRegistry histos{"histos", {}, OutputObjHandlingPolicy::AnalysisObject};

  // Defining a configurable for the bin axis:
  Configurable<int> nBinsPt{"nBinsPt", 100, "Nbins in pT histo"}; // This has to be declared with braces, not parenthesis.
  // This comes from uniform inicialization in C++11. It helps to make the initialization non-ambiguous: it is a template declaration, not a function declaration.

  void init(InitContext const&)
  {
    // define axes you want to use
    const AxisSpec axisEta{30, -1.5, +1.5, "#eta"};
    const AxisSpec axisPt{nBinsPt, 0, 10, "p_{T}"}; // Defined as const because this will not change!
    const AxisSpec axisEv{1, 0, 1, "Counting_ax"}; // has to be defined encapsulated by braces!

    // create histograms
    histos.add("etaHistogram", "etaHistogram", kTH1F, {axisEta});
    histos.add("ptHistogram", "ptHistogram", kTH1F, {axisPt}); // Used the kTH1F enumerator to construct the histogram as a TH1F
    histos.add("hEventCounter", "hEventCounter", kTH1I, {axisEv});
  }

    // Deleting the old process to have an event-focused task, not the earlier "loop through all events" task:
  // void process(aod::TracksIU const& tracks)
  // {
  //   for (auto& track : tracks) { // Will loop on the tracks iterable with the track iterator
  //     histos.fill(HIST("etaHistogram"), track.eta());
  //     histos.fill(HIST("ptHistogram"), track.pt());
  //   }
  // }

    // Declaring the new process:
    // It will access collision level information from the aod table.
    // The first argument is something like the iterator that we will use: we will iterate through collisions.
    // (it is not an iterator object per se, though. It only tells O2 what we will iterate on.)
  void process(aod::Collision const& collision, soa::Join<aod::Tracks, aod::TracksExtra, aod::TracksDCA> const& tracks){
      // Counting the number of different collisions/events:
    histos.fill(HIST("hEventCounter"), 0.5);
    // Now we introduce a loop on all available tracks, divided by collision:
      // Will loop on the tracks iterable with the track iterator. Notice there is NO NEED to use const& track,
      // as tracks is already const&, then its iterators will also be const objects unless explicitly changed!
      // Will declare it explicitly though, for better readability
    for (auto const& track : tracks){
      // Variables used here: tpcNClsCrossedRows ("Number of crossed TPC Rows"), which is the number of the TPC layers that the track has crossed.
      // more is better, for better resolution!
      // dcaXY, which gives us how far the track origin would be from the primary vertex of this collision.
      // Interestingly, this variable is associated implicitly to the current collision through the table joining!
      // Thus it already refers to the current collision object of the AOD!
      
        // Selecting with these ifs is not the best option. We could use filters or partitions!
        // But for tutorial purposes this is OK (for now...)
          // This REALLY increases the runtime!
      if(track.tpcNClsCrossedRows() < 70) continue;
      if(fabs(track.dcaXY()) > 0.2) continue;
      // Have to use fabs here, because we are dealing with operations in the O2 soa (structure of arrays) tables
      // Use fabs instead of abs, cause some compilers can convert it to int. Or use std::abs().
      // fabs is preferred, though.

      // Then fill the other histograms we were already studying:
      histos.fill(HIST("etaHistogram"), track.eta());
      histos.fill(HIST("ptHistogram"), track.pt());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<myExampleTask2>(cfgc)};
}
