import React from 'react';
import { useControls, useCreateStore } from 'leva';
import { Blendshape } from '../types/types';

interface Props {
  blendshapes: Blendshape[];
  setWeights: (w: number[]) => void;
  store: ReturnType<typeof useCreateStore>;
}

export const BlendshapeSliders: React.FC<Props> = ({ blendshapes, setWeights, store }) => {
  // Create initial values object
  const initialValues = React.useMemo(() => {
    const values: { [key: string]: number } = {};
    blendshapes.forEach(bs => {
      values[bs.name] = 0;
    });
    return values;
  }, [blendshapes]);

  // Create the controls configuration
  const config = React.useMemo(() => {
    return blendshapes.reduce((acc, bs) => ({
      ...acc,
      [bs.name]: {
        value: initialValues[bs.name],
        min: 0,
        max: 1,
        step: 0.01,
      }
    }), {});
  }, [blendshapes, initialValues]);

  // Use controls with the configuration
  const values = useControls(config, { store });

  // Update weights when values change
  React.useEffect(() => {
    if (!values || Object.keys(values).length === 0) return;

    const newWeights = blendshapes.map(bs => {
      const value = values[bs.name];
      return value ?? 0;
    });

    console.log('Setting weights:', newWeights);
    setWeights(newWeights);
  }, [values, blendshapes, setWeights]);

  return null;
}; 