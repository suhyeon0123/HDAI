import os
import csv
import array
import base64
import xmltodict
import numpy as np

class Reader:
    """ Extract voltage data from a ECG XML file """
    def __init__(self, path, augmentLeads=False):
            with open(path, 'rb') as xml:
                self.ECG = xmltodict.parse(xml.read().decode('utf8'))
            
            self.augmentLeads           = augmentLeads
            self.path                   = path

            self.PatientDemographics    = self.ECG['RestingECG']['PatientDemographics']
            self.Waveforms              = self.ECG['RestingECG']['Waveform']

            self.LeadVoltages           = self.makeLeadVoltages()

    
    def makeLeadVoltages(self):

        num_leads = 0

        leads = {}

        
        self.count = len(self.Waveforms)
        return
        for lead in self.Waveforms['LeadData']:
            num_leads += 1
            
            lead_data = lead['WaveFormData']
            lead_b64  = base64.b64decode(lead_data)
            lead_vals = np.array(array.array('h', lead_b64))

            leads[ lead['LeadID'] ] = lead_vals
        
        if num_leads == 8 and self.augmentLeads:

            leads['III'] = np.subtract(leads['II'], leads['I'])
            leads['aVR'] = np.add(leads['I'], leads['II'])*(-0.5)
            leads['aVL'] = np.subtract(leads['I'], 0.5*leads['II'])
            leads['aVF'] = np.subtract(leads['II'], 0.5*leads['I'])
        
        return leads

    def getLeadVoltages(self, LeadID):
        return self.LeadVoltages[LeadID]
    
    def getAllVoltages(self):
        return self.LeadVoltages
    
    def getCount(self):
        return self.count


def main():
    path = 'data/data/train/arrhythmia/5_2_000436_ecg.xml'
    ecg = Reader(path, augmentLeads=True)
    print(ecg.getCount())
    ecg.getAllVoltages()
    path = 'data/data/validation/arrhythmia/8_2_009000_ecg.xml'
    ecg = Reader(path, augmentLeads=True)
    print(ecg.getCount())
    ecg.getAllVoltages()

if __name__ == '__main__':
    main()