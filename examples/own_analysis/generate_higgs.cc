// Small real-shower interoperability sample, not a precision production.
#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC3.h"
#include <fstream>
#include <iomanip>
#include <stdexcept>
int main(int argc, char** argv) {
  if (argc != 4) throw std::runtime_error("generate_higgs OUTPUT.hepmc EVENTS SEED");
  const int wanted=std::stoi(argv[2]), seed=std::stoi(argv[3]);
  if (wanted<1 || seed<1 || seed>900000000) throw std::runtime_error("invalid count or seed");
  Pythia8::Pythia p;
  for (const auto& setting : {"Beams:idA = 2212", "Beams:idB = 2212", "Beams:eCM = 13000",
      "HiggsSM:gg2H = on", "25:m0 = 125.0", "25:mWidth = 0.0", "25:onMode = off",
      "25:onIfMatch = 22 22", "TimeShower:QEDshowerByGamma = off",
      "Tune:pp = 14", "Random:setSeed = on", "Next:numberShowEvent = 0",
      "Next:numberShowInfo = 0", "Next:numberShowProcess = 0", "Next:numberCount = 0"})
    if (!p.readString(setting)) throw std::runtime_error(setting);
  p.readString("Random:seed = "+std::to_string(seed));
  if (!p.init()) throw std::runtime_error("Pythia initialization failed");
  HepMC3::Pythia8ToHepMC3 convert;
  auto run=std::make_shared<HepMC3::GenRunInfo>();
  run->set_weight_names({"original_generator"});
  run->add_attribute("sample", std::make_shared<HepMC3::StringAttribute>(
      "Real Pythia gg->H->gamma gamma shower; small interoperability example; seed="+std::to_string(seed)));
  HepMC3::WriterAscii writer(argv[1],run);
  writer.set_precision(17);
  int done=0, failures=0;
  while (done<wanted) {
    if (!p.next()) { if (++failures>100) throw std::runtime_error("too many failed events"); continue; }
    HepMC3::GenEvent event(HepMC3::Units::GEV,HepMC3::Units::MM);
    if (!convert.fill_next_event(p,&event,done)) throw std::runtime_error("HepMC conversion failed");
    event.set_run_info(run);
    writer.write_event(event);
    ++done;
  }
  writer.close();
  std::ofstream summary(std::string(argv[1])+".json");
  summary << std::setprecision(17) << "{\"pythia_version\":" << PYTHIA_VERSION
          << ",\"events\":" << done << ",\"seed\":" << seed
          << ",\"failed_attempts\":" << failures << ",\"generator_sigma_pb\":" << p.info.sigmaGen()*1e9
          << ",\"note\":\"Generator metadata is retained; fitted rates come from released NNLO inputs.\"}\n";
  p.stat();
}
