// Genuine Rivet analysis. Named weights are multiplexed by Rivet itself.
#include "Rivet/Analysis.hh"
#include "Rivet/Projections/FinalState.hh"
#include <set>
namespace Rivet {
class MC_MAXENT_HGG : public Analysis {
public:
  explicit MC_MAXENT_HGG(bool normalize=false) : Analysis("MC_MAXENT_HGG"), _normalize(normalize) {}
  void init() {
    declare(FinalState(Cuts::pid==22),"Photons");
    book(_all,"all"); book(_fid,"fiducial"); book(_cut,"fiducial_pt45");
    book(_pt,"fiducial_pt",std::vector<double>{0,20,45,80,150,300,13000});
  }
  void analyze(const Event& event) {
    Particles photons;
    for (const Particle& p : apply<FinalState>(event,"Photons").particles()) {
      bool fromH=false;
      Particles pending=p.parents(); std::set<int> seen;
      while (!pending.empty()) {
        const Particle ancestor=pending.back(); pending.pop_back();
        if (!seen.insert(ancestor.genParticle()->id()).second) continue;
        if (ancestor.pid()==25) { fromH=true; break; }
        if (ancestor.pid()==22) {
          const auto parents=ancestor.parents();
          pending.insert(pending.end(),parents.begin(),parents.end());
        }
      }
      if (fromH) photons.push_back(p);
    }
    if (photons.size()!=2) throw Error("expected exactly two Higgs-descended final photons");
    photons = sortByPt(photons);
    const FourMomentum diphoton=photons[0].momentum()+photons[1].momentum();
    const double mass=diphoton.mass()/GeV, pt=diphoton.pT()/GeV;
    _all->fill();
    bool fid=105<mass && mass<160 && photons[0].pT()/GeV>.35*mass && photons[1].pT()/GeV>.25*mass;
    for (const auto& p : photons) {
      const double eta=p.abseta();
      fid=fid && eta<2.37 && !(1.37<eta && eta<1.52);
    }
    if (!fid) return;
    _fid->fill(); _pt->fill(pt);
    if (pt>45) _cut->fill();
  }
  void finalize() {
    // Absolute pb/event vectors are already cross-section contributions.
    // Conventional normalization is also tested, with the rate for THIS stream.
    if (_normalize) {
      const double factor=crossSection()/picobarn/sumOfWeights();
      scale(_all,factor); scale(_fid,factor); scale(_cut,factor); scale(_pt,factor);
    }
  }
private:
  bool _normalize;
  CounterPtr _all,_fid,_cut;
  Histo1DPtr _pt;
};
}
