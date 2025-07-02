import React from 'react';
import { Slider, Text, Stack, Alert } from '@mantine/core';
import { IconAlertCircle } from '@tabler/icons-react';

interface Blendshape {
  name: string;
  vertices: Float32Array;
  center: THREE.Vector3;
  normal: THREE.Vector3;
  maxDisplacement: number;
}

interface BlendshapeControlsProps {
  blendshapes: Blendshape[];
  weights: number[];
  setWeights: (weights: number[]) => void;
  loading: boolean;
  error: string | null;
}

const BlendshapeControls: React.FC<BlendshapeControlsProps> = ({
  blendshapes,
  weights,
  setWeights,
  loading,
  error
}) => {
  const handleWeightChange = (index: number, value: number) => {
    const newWeights = [...weights];
    newWeights[index] = value;
    setWeights(newWeights);
  };

  if (loading) {
    return <Text>Loading blendshapes...</Text>;
  }

  if (error) {
    return (
      <Alert icon={<IconAlertCircle size={16} />} title="Error" color="red">
        {error}
      </Alert>
    );
  }

  return (
    <Stack>
      <Text fw={500} size="lg" style={{ marginBottom: '1rem' }}>
        Blendshape Controls
      </Text>
      {blendshapes.map((blendshape, index) => (
        <div key={blendshape.name}>
          <Text size="sm" style={{ marginBottom: '0.25rem' }}>
            {blendshape.name.replace('faceMuscles.', '')}
          </Text>
          <Slider
            value={weights[index]}
            onChange={(value) => handleWeightChange(index, value)}
            min={0}
            max={1}
            step={0.01}
            label={(value) => value.toFixed(2)}
            styles={{
              root: { marginBottom: 20 },
              track: { backgroundColor: '#E9ECEF' },
              bar: { backgroundColor: '#228BE6' },
              thumb: { 
                backgroundColor: '#FFFFFF',
                borderColor: '#228BE6',
                borderWidth: 2
              },
              label: { 
                backgroundColor: '#228BE6',
                color: 'white'
              }
            }}
          />
        </div>
      ))}
    </Stack>
  );
};

export default BlendshapeControls; 