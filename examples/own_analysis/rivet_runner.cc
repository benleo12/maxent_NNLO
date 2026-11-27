// Uses the real Rivet library; also works when a legacy rivet Python launcher is unavailable.
#include "Rivet/AnalysisHandler.hh"
#include "HepMC3/ReaderAscii.h"
#include "MC_MAXENT_HGG.cc"
#include <fstream>
#include "YODA/WriterYODA.h"
#include <map>
int main(int argc,char** argv) {
  if (argc!=4 && argc!=5) throw std::runtime_error("rivet_runner INPUT OUTPUT absolute|normalized [RATES.tsv]");
  const std::string mode=argv[3];
  if (mode!="absolute" && mode!="normalized") throw std::runtime_error("unknown normalization mode");
  std::map<std::string,double> rates;
  if (argc==5) { std::ifstream f(argv[4]); std::string name; double rate; while(f>>name>>rate) rates[name]=rate; }
  HepMC3::ReaderAscii reader(argv[1]);
  Rivet::AnalysisHandler analysis;
  analysis.addAnalysis(new Rivet::MC_MAXENT_HGG(mode=="normalized"));
  std::size_t n=0;
  for (;;) {
    HepMC3::GenEvent event;
    reader.read_event(event);
    if (reader.failed()) break;
    if (n==0) {
      if (!event.weight_names().empty()) analysis.setNominalWeightName(event.weight_names().front());
      analysis.init(event);
      if (!rates.empty()) {
        std::vector<std::pair<double,double>> crossSections;
        for (std::size_t i=0; i<event.weight_names().size(); ++i) {
          const auto& name=event.weight_names()[i];
          // Only explicitly named streams are overridden in the negative control.
          // Original generator streams retain their own cross-section metadata.
          if (rates.count(name)) crossSections.emplace_back(rates.at(name),0.0);
          else crossSections.emplace_back(event.cross_section()->xsec(i),event.cross_section()->xsec_err(i));
        }
        // Explicit per-member rate, not a rescaled nominal rate.
        analysis.setCrossSection(crossSections,true);
      }
    }
    analysis.analyze(event); ++n;
  }
  if (!n) throw std::runtime_error("empty input");
  analysis.finalize();
  auto& writer=YODA::WriterYODA::create(); writer.setPrecision(17);
  writer.write(argv[2],analysis.getYodaAOs());
  std::cout<<"Rivet processed "<<n<<" records in "<<mode<<" mode\n";
}
