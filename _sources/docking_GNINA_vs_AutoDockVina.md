# Example 4 - Different Choices of Surrogate for Docking Inference

To run this notebook you need to have [GNINA](https://github.com/gnina/gnina) and [AutoDock Vina via DockString](https://github.com/dockstring/dockstring) installed. GNINA can be downloaded as a linux binary and made executable (see this [notebook](https://colab.research.google.com/drive/1QYo5QLUE80N_G28PlpYs6OKGddhhd931?usp=sharing) for installation).


```python
from rdkit import Chem
from rdkit.Chem import AllChem
import subprocess

def predictGNINA_docking(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol)
    AllChem.UFFOptimizeMolecule(mol)
    
    # Save to PDB file (required format for GNINA)
    output_file = "molecule.pdb"
    Chem.MolToPDBFile(mol, output_file)

     # Command to run GNINA
    command = "../dziner/surrogates/gnina -r ../dziner/surrogates/Binding/WDR5_target.pdbqt -l molecule.pdb --autobox_ligand ../dziner/surrogates/Binding/WDR5_target.pdbqt --seed 0"
    
    # Run the command and capture the output
    result = subprocess.getoutput(command)
    
    # Initialize a list to store affinity values
    affinities = []
    
    # Loop through the output lines to find affinity scores
    for line in result.splitlines():
        # Skip lines that don't start with a mode index (i.e., a number)
        if line.strip() and line.split()[0].isdigit():
            # Split the line and extract the affinity (2nd column)
            try:
                affinity = float(line.split()[1])
                affinities.append(affinity)
            except ValueError:
                # Skip lines that don't have a valid number in the second column
                continue
    
    # Find the minimum affinity score
    if affinities:
        best_docking_score = min(affinities)
        # print(f"Best docking score: {best_docking_score}")
    else:
        # print("No valid docking scores found.")
        best_docking_score = None
    return best_docking_score
```


```python
from dockstring import load_target

def predictAutoVina_docking(smiles):
    '''
    This tool predicts the docking score to WDR5 for a SMILES molecule. Lower docking score means a larger binding affinity.
    This model based on Autodock Vina from this paper: https://pubs.acs.org/doi/10.1021/acs.jcim.1c01334 
    '''
    smiles = smiles.replace("\n", "").replace("'", "")
    target = load_target("WDR5")
    try:
        score, affinities = target.dock(smiles)
    except:
        score = 'Invalid molecule'
    return score


```

## GNINA vs AutoDock Vina


```python
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

df = pd.read_csv('../paper/results/1.binding-WDR5/new_candidate/anthropic/WDR5-20-it(0).csv', usecols=['Iteration', 'SMILES'] )
df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Iteration</th>
      <th>SMILES</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>CN1CCN(C2=C(NC(C3=CC=CC=C3Cl)=O)C=C([N+]([O-])...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>CN1CCN(C2=C(NC(C3=CC=CC=C3F)=O)C=C([N+]([O-])=...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>CN1CCN(C2=C(NC(C3=CC=CC4=CC=CC=C43)=O)C=C([N+]...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>CN1CCN(C2=C(NC(C3=CC=CC4=CC=CC=C43)=O)C=C(C#N)...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>CN1CCN(C2=C(NC(C3=CC=CC4=CC=CC=C43)=O)C=C(C(=O...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>CN1CCN(C2=C(NC(C3=CC=CC4=CC=CC=C43)=O)C=C(C(=O...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>CN1CCN(CC(=O)N)CC1C2=C(NC(C3=CC=CC4=CC=CC=C43)...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>CN1CCN(C2=C(NC(C3=C(F)C=CC4=CC=CC=C43)=O)C=C(C...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>CN1CCN(C2=C(NC(=O)CC3=CC=CC4=CC=CC=C43)C=C(C(=...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>CN1CCN(CC2=CC=C(NC(C3=CC=CC4=CC=CC=C43)=O)C(C(...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>CN1CCN(C2=C(NC(C3=CC=CC4=CC=CC=C43)=O)C=C(C(=O...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>CCN1CCN(C2=C(NC(C3=CC=CC4=CC=CC=C43)=O)C=C(C(=...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>CN1CCN(C2=C(NC(C3=CC=CC4=CC=CC=C43)=O)C(F)=C(C...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>CN1CCN(C2=C(NC(=N)C3=CC=CC4=CC=CC=C43)C=C(C(=O...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>CN1CCN(C2=C(NC(C3=CC=CC4=CC=CC=C43)=O)C=C(C(=O...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>CN1CCN(C2=C(NC(C3=CC=CC4=CC=CC=C43)=O)C=C(C(=O...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>CN1CCN(C2=C(NC(C3=CC=C(F)C4=CC=CC=C43)=O)C=C(C...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>CN1CCN(C2=C(NC(C3=CC=C(Cl)C4=CC=CC=C43)=O)C=C(...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>CCN1CCN(C2=C(NC(C3=CC=C(Cl)C4=CC=CC=C43)=O)C=C...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>CN1CCN(C2=C(NC(C3=CC=C(Cl)C4=CC=CC=C43)=O)C=C(...</td>
    </tr>
  </tbody>
</table>
</div>




```python
%%time ## 22.65 seconds/ molecule
df['Docking Score (GNINA)'] = df['SMILES'].progress_apply(predictGNINA_docking)
```

    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [07:33<00:00, 22.65s/it]

    CPU times: user 2.16 s, sys: 58.6 ms, total: 2.22 s
    Wall time: 7min 33s


    



```python
%%time ## 21.4 seconds/ molecule
df['Docking Score (AutoDock Vina)'] = df['SMILES'].progress_apply(predictAutoVina_docking)
```

    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20/20 [07:08<00:00, 21.42s/it]

    CPU times: user 2.91 s, sys: 134 ms, total: 3.04 s
    Wall time: 7min 8s


    



```python
import numpy as np

df['\u0394'] = np.abs(df['Docking Score (GNINA)']- df['Docking Score (AutoDock Vina)'])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Iteration</th>
      <th>SMILES</th>
      <th>Docking Score (GNINA)</th>
      <th>Docking Score (AutoDock Vina)</th>
      <th>Δ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>CN1CCN(C2=C(NC(C3=CC=CC=C3Cl)=O)C=C([N+]([O-])...</td>
      <td>-6.33</td>
      <td>-6.7</td>
      <td>0.37</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>CN1CCN(C2=C(NC(C3=CC=CC=C3F)=O)C=C([N+]([O-])=...</td>
      <td>-5.52</td>
      <td>-6.6</td>
      <td>1.08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>CN1CCN(C2=C(NC(C3=CC=CC4=CC=CC=C43)=O)C=C([N+]...</td>
      <td>-7.55</td>
      <td>-7.6</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>CN1CCN(C2=C(NC(C3=CC=CC4=CC=CC=C43)=O)C=C(C#N)...</td>
      <td>-7.83</td>
      <td>-7.6</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>CN1CCN(C2=C(NC(C3=CC=CC4=CC=CC=C43)=O)C=C(C(=O...</td>
      <td>-8.26</td>
      <td>-8.4</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>CN1CCN(C2=C(NC(C3=CC=CC4=CC=CC=C43)=O)C=C(C(=O...</td>
      <td>-8.22</td>
      <td>-8.3</td>
      <td>0.08</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>CN1CCN(CC(=O)N)CC1C2=C(NC(C3=CC=CC4=CC=CC=C43)...</td>
      <td>-8.88</td>
      <td>-7.9</td>
      <td>0.98</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>CN1CCN(C2=C(NC(C3=C(F)C=CC4=CC=CC=C43)=O)C=C(C...</td>
      <td>-8.15</td>
      <td>-8.1</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>CN1CCN(C2=C(NC(=O)CC3=CC=CC4=CC=CC=C43)C=C(C(=...</td>
      <td>-7.82</td>
      <td>-8.2</td>
      <td>0.38</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>CN1CCN(CC2=CC=C(NC(C3=CC=CC4=CC=CC=C43)=O)C(C(...</td>
      <td>-7.36</td>
      <td>-8.2</td>
      <td>0.84</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>CN1CCN(C2=C(NC(C3=CC=CC4=CC=CC=C43)=O)C=C(C(=O...</td>
      <td>-8.35</td>
      <td>-8.2</td>
      <td>0.15</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>CCN1CCN(C2=C(NC(C3=CC=CC4=CC=CC=C43)=O)C=C(C(=...</td>
      <td>-8.07</td>
      <td>-7.9</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>CN1CCN(C2=C(NC(C3=CC=CC4=CC=CC=C43)=O)C(F)=C(C...</td>
      <td>-7.55</td>
      <td>-8.0</td>
      <td>0.45</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>CN1CCN(C2=C(NC(=N)C3=CC=CC4=CC=CC=C43)C=C(C(=O...</td>
      <td>-7.96</td>
      <td>-7.9</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>CN1CCN(C2=C(NC(C3=CC=CC4=CC=CC=C43)=O)C=C(C(=O...</td>
      <td>-9.02</td>
      <td>-8.7</td>
      <td>0.32</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>CN1CCN(C2=C(NC(C3=CC=CC4=CC=CC=C43)=O)C=C(C(=O...</td>
      <td>-8.14</td>
      <td>-8.6</td>
      <td>0.46</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>CN1CCN(C2=C(NC(C3=CC=C(F)C4=CC=CC=C43)=O)C=C(C...</td>
      <td>-8.58</td>
      <td>-8.8</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>CN1CCN(C2=C(NC(C3=CC=C(Cl)C4=CC=CC=C43)=O)C=C(...</td>
      <td>-8.91</td>
      <td>-8.9</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>CCN1CCN(C2=C(NC(C3=CC=C(Cl)C4=CC=CC=C43)=O)C=C...</td>
      <td>-7.51</td>
      <td>-8.7</td>
      <td>1.19</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>CN1CCN(C2=C(NC(C3=CC=C(Cl)C4=CC=CC=C43)=O)C=C(...</td>
      <td>-8.68</td>
      <td>-8.3</td>
      <td>0.38</td>
    </tr>
  </tbody>
</table>
</div>


