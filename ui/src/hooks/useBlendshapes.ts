import { useState, useEffect } from "react";
import * as THREE from "three";

interface Blendshape {
  name: string;
  vertices: Float32Array;
  center: THREE.Vector3;
  normal: THREE.Vector3;
  maxDisplacement: number;
}

interface FaceData {
  baseVertices: number[];
  baseFaces: number[];
  numVertices: number;
  numFaces: number;
  blendshapes: {
    name: string;
    vertices: number[];
    center: [number, number, number];
    normal: [number, number, number];
    maxDisplacement: number;
  }[];
}

export const useBlendshapes = () => {
  const [blendshapes, setBlendshapes] = useState<Blendshape[]>([]);
  const [weights, setWeights] = useState<number[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [baseVertices, setBaseVertices] = useState<Float32Array>(
    new Float32Array(),
  );
  const [baseFaces, setBaseFaces] = useState<Uint32Array>(new Uint32Array());

  useEffect(() => {
    const loadBlendshapes = async () => {
      try {
        // Fetch face data from the Python backend
        const response = await fetch("http://localhost:5000/face-data");
        if (!response.ok) {
          throw new Error("Failed to load face data");
        }

        const data: FaceData = await response.json();

        // Validate data
        if (
          !data.baseVertices ||
          !data.baseFaces ||
          !data.blendshapes ||
          !data.numVertices ||
          !data.numFaces
        ) {
          throw new Error("Invalid data format received from server");
        }

        // Log received data for debugging
        console.log("Received data:", {
          vertexCount: data.numVertices,
          faceCount: data.numFaces,
          blendshapeCount: data.blendshapes.length,
          sampleVertices: data.baseVertices.slice(0, 9),
          sampleFaces: data.baseFaces.slice(0, 9),
        });

        // Check for invalid values with detailed logging
        const invalidVertices = data.baseVertices
          .map((v, i) => ({ value: v, index: i }))
          .filter(({ value }) => {
            // Check if it's an array of 3 numbers
            if (!Array.isArray(value) || value.length !== 3) {
              return true;
            }
            // Check if all components are finite numbers
            return value.some((component) => !Number.isFinite(component));
          });

        if (invalidVertices.length > 0) {
          console.error("Found invalid vertices:", {
            count: invalidVertices.length,
            first10Invalid: invalidVertices.slice(0, 10),
            totalVertices: data.baseVertices.length,
          });
          throw new Error("Invalid vertex data received");
        }

        const invalidFaces = data.baseFaces
          .map((f, i) => ({ value: f, index: i }))
          .filter(({ value }) => {
            // Check if it's an array of 3 indices
            if (!Array.isArray(value) || value.length !== 3) {
              return true;
            }
            // Check if all indices are valid integers within vertex range
            return value.some(
              (index) =>
                !Number.isInteger(index) ||
                index < 0 ||
                index >= data.numVertices,
            );
          });

        if (invalidFaces.length > 0) {
          console.error("Found invalid faces:", {
            count: invalidFaces.length,
            first10Invalid: invalidFaces.slice(0, 10),
            totalFaces: data.baseFaces.length,
          });
          throw new Error("Invalid face indices received");
        }

        // Convert arrays to typed arrays - flatten the arrays first
        const vertices = new Float32Array(data.baseVertices.flat());
        const faces = new Uint32Array(data.baseFaces.flat());

        // Validate array lengths
        if (vertices.length !== data.numVertices * 3) {
          throw new Error(
            `Vertex data length mismatch: expected ${data.numVertices * 3}, got ${vertices.length}`,
          );
        }
        if (faces.length !== data.numFaces * 3) {
          throw new Error(
            `Face data length mismatch: expected ${data.numFaces * 3}, got ${faces.length}`,
          );
        }

        // Set base vertices and faces
        setBaseVertices(vertices);
        setBaseFaces(faces);

        // Convert the data to the format we need
        const convertedBlendshapes: Blendshape[] = data.blendshapes.map(
          (bs) => ({
            name: bs.name,
            vertices: new Float32Array(bs.vertices),
            center: new THREE.Vector3(...bs.center),
            normal: new THREE.Vector3(...bs.normal),
            maxDisplacement: bs.maxDisplacement,
          }),
        );

        // Initialize weights array
        const initialWeights = new Array(data.blendshapes.length).fill(0);

        setBlendshapes(convertedBlendshapes);
        setWeights(initialWeights);
        setLoading(false);
        setError(null);
      } catch (err) {
        console.error("Error loading blendshapes:", err);
        setError(err instanceof Error ? err.message : "Unknown error");
        setLoading(false);
      }
    };

    loadBlendshapes();
  }, []);

  return {
    blendshapes,
    weights,
    setWeights,
    loading,
    error,
    baseVertices,
    baseFaces,
  };
};

