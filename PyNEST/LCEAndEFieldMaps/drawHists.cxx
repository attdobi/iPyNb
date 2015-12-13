#include <iostream>
#include "TMath.h"
#include "TH2D.h"
#include "TFile.h"
#include "TCanvas.h"
#include <fstream>
#include "TLatex.h"

using namespace std;

/*Macro to create .root files with histogram maps from text files given*/

void drawEGraph(Bool_t save = 0){
  ifstream infile;
  infile.open("EField.txt");
  double x,y,z,a,b;
  double nR = 36;
  double nZ = 872;
  double rMin = 75.825;
  double zMin = -15.875;
  double rMax = rMin+nR*0.2;
  double zMax = zMin+nZ*0.2;
  cout << rMin+nR*0.2 << "\t" << zMin+nZ*0.2 << endl;
  TH2D* eFieldMap = new TH2D("E","E Field",nR,rMin,rMax,nZ,zMin,zMax);
  double rBinWidth = eFieldMap->GetXaxis()->GetBinWidth(1);
  double zBinWidth = eFieldMap->GetYaxis()->GetBinWidth(1);
  cout << rBinWidth << "\t" << zBinWidth << endl;
  while(!infile.eof()){
    infile >> x >> y >> z;
    a = TMath::Nint((x-rMin-0.1)/rBinWidth)+1;
    b = TMath::Nint((y-zMin-0.1)/zBinWidth)+1;
    if(a<nR && b<nZ)
      eFieldMap->SetBinContent(a,b,z/1000);
  }
  infile.close();
  eFieldMap->GetZaxis()->SetRangeUser(0,65);
  eFieldMap->SetStats(0);
  eFieldMap->GetXaxis()->SetTitle("r [cm]");
  eFieldMap->GetYaxis()->SetTitle("z [cm]");
  TCanvas* c1 = new TCanvas();
  eFieldMap->Draw("colz");
  TLatex* t = new TLatex(.89,.91,"kV/cm");
  t->SetTextSize(0.045);
  t->SetNDC(kTRUE);
  t->Draw();
  if(save)
    c1->Print("skinEfield.root");
}

void drawLCEGraph(Bool_t save = 0){
  ifstream infile;
  infile.open("LCE.txt");
  double x,y,z;
  double rMin = 0;
  double zMin = -50;
  double nR = 84;
  double nZ = 200;
  TH2D* lceMap = new TH2D("LCE","LCE",nR,rMin,rMin+nR,nZ,zMin,zMin+nZ);
  double rBinWidth = lceMap->GetXaxis()->GetBinWidth(1);
  double zBinWidth = lceMap->GetYaxis()->GetBinWidth(1);
  cout << rBinWidth << "\t" << zBinWidth << endl;
  while(!infile.eof()){
    infile >> x >> y >> z;
    lceMap->SetBinContent((x-rMin-0.5)/rBinWidth,(y-zMin-0.5)/zBinWidth,z); //Fill also fine
  }
  infile.close();
  TCanvas* c1 = new TCanvas();
  lceMap->SetStats(0);
  lceMap->GetXaxis()->SetTitle("r [cm]");
  lceMap->GetYaxis()->SetTitle("z [cm]");
  lceMap->GetZaxis()->SetLabelSize(0.03);
  lceMap->Draw("colz");
  if(save)
    c1->Print("skinLCEwithPMT.root");
}

