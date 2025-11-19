import React, { useState } from 'react';
import './ReasoningDisplay.css';

const ReasoningDisplay = ({ steps, structured, unstructured }) => {
    const [isExpanded, setIsExpanded] = useState(false);

    if (!steps && !structured && !unstructured) return null;

    return (
        <div className="reasoning-display">
            <button
                className="reasoning-toggle"
                onClick={() => setIsExpanded(!isExpanded)}
            >
                <span className="toggle-icon">{isExpanded ? '▼' : '▶'}</span>
                <span className="toggle-text">
                    Reasoning & Sources {steps ? `(${steps} steps)` : ''}
                </span>
            </button>

            {isExpanded && (
                <div className="reasoning-content">
                    {structured && (
                        <div className="source-section">
                            <h4>Structured Data (EMR)</h4>
                            {Array.isArray(structured) ? (
                                <ul>
                                    {structured.map((item, idx) => (
                                        <li key={idx}>{item}</li>
                                    ))}
                                </ul>
                            ) : (
                                <p>{JSON.stringify(structured)}</p>
                            )}
                        </div>
                    )}

                    {unstructured && (
                        <div className="source-section">
                            <h4>Medical Knowledge</h4>
                            {Array.isArray(unstructured) ? (
                                <ul>
                                    {unstructured.map((item, idx) => (
                                        <li key={idx}>{item}</li>
                                    ))}
                                </ul>
                            ) : (
                                <p>{JSON.stringify(unstructured)}</p>
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default ReasoningDisplay;
