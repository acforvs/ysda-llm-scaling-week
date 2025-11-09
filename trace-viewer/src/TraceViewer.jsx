import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import hljs from 'highlight.js';
import 'highlight.js/styles/github.css';
import { getLast } from './utils';
import { marked } from 'marked';

function TraceViewer() {
  // Parse URL params
  const urlParams = new URLSearchParams(window.location.search)
  const tracePath = urlParams.get('trace');  // JSON file that has everything
  const targetSourcePath = urlParams.get('source');  // Source file to display
  const targetLineNumber = parseInt(urlParams.get('line')) || null;  // Line number to highlight
  const targetStepIndex = parseInt(urlParams.get('step')) || null;  // Step index to highlight
  const rawMode = urlParams.get('raw');
  const animateMode = urlParams.get('animate');
  const navigate = useNavigate();

  const [error, setError] = useState(null);
  const [trace, setTrace] = useState(null);

  // Add new state for position and offset
  const [envPosition, setEnvPosition] = useState({ x: 20, y: 20 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });

  // Fetch trace from backend
  useEffect(() => {
    if (!tracePath) {
      return;
    }

    const fetchData = async () => {
      try {
        const url = tracePath;
        const response = await axios.get(url);
        setTrace(response.data);
        const basePath = tracePath.split('/').pop().replace('.json', '');
        document.title = `Trace - ${basePath}`;
      } catch (error) {
        console.error(error);
        setError(error.message);
      }
    };
    fetchData();
  }, [tracePath]);

  // Add keyboard navigation
  useEffect(() => {
    if (!trace) {
      return;
    }

    const handleKeyDown = (event) => {
      if (event.altKey || event.ctrlKey) {  // Don't capture alt-right (for web page navigation)
        return;
      }

      if (!event.shiftKey && (event.key === 'ArrowRight' || event.key === 'l')) {
        stepForward({trace, currentStepIndex, navigate});
      } else if (!event.shiftKey && (event.key === 'ArrowLeft' || event.key === 'h')) {
        stepBackward({currentStepIndex, navigate});
      } else if ((event.shiftKey && event.key === 'ArrowRight') || event.key === 'j') {
        stepOverForward({trace, currentStepIndex, navigate});
      } else if ((event.shiftKey && event.key === 'ArrowLeft') || event.key === 'k') {
        stepOverBackward({trace, currentStepIndex, navigate});
      } else if (event.shiftKey && (event.key === 'ArrowRight' || event.key === 'l')) {
        stepForward({trace, currentStepIndex, navigate, stayOnSameLine: true});
      } else if (event.key === 'u') {
        stepUp({trace, currentStepIndex, navigate});
      } else if (event.key === 'R') {
        toggleRawMode({rawMode, navigate});
      } else if (event.key === 'A') {
        toggleAnimateMode({animateMode, navigate});
      } else if (event.key === 'g') {
        gotoTrace({tracePath, navigate});
      } else {
        return;
      }
      // Applies to any key event that we've handled
      event.preventDefault();
      event.stopPropagation();
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [trace, targetStepIndex, targetLineNumber, rawMode, animateMode, navigate]);

  // Update drag handlers
  const handleMouseDown = (e) => {
    if (e.target.closest('.env-panel')) {
      const panel = e.target.closest('.env-panel');
      const rect = panel.getBoundingClientRect();
      setDragOffset({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top
      });
      setIsDragging(true);
      e.preventDefault();
    }
  };

  const handleMouseMove = (e) => {
    if (isDragging) {
      setEnvPosition({
        x: e.clientX - dragOffset.x,
        y: e.clientY - dragOffset.y
      });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Add event listeners for dragging
  useEffect(() => {
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging]);

  // Not ready
  if (!tracePath) {
    return <div>No trace path provided</div>;
  }
  if (error) {
    return renderError(error);
  }
  if (!trace) {
    return "Loading...";
  }

  // Figure out the current line number and step index from each other
  let currentStepIndex;
  let currentStackElement;
  let currentStep;
  let currentLineNumber;
  let currentPath;
  if (targetStepIndex === null && targetLineNumber === null) {
    // Default to the first step if nothing is specified
    currentStepIndex = 0;
    currentStep = trace.steps[currentStepIndex];
    currentStackElement = getLast(currentStep.stack);
    currentPath = currentStackElement.path;
    currentLineNumber = currentStackElement.line_number;
  } else if (targetStepIndex !== null) {
    currentStepIndex = targetStepIndex;
    // Find the line number with the target step index
    currentStep = trace.steps[currentStepIndex];
    currentStackElement = getLast(currentStep.stack);
    currentPath = currentStackElement.path;
    currentLineNumber = currentStackElement.line_number;
  } else {
    currentPath = targetSourcePath;
    currentLineNumber = targetLineNumber;
    // Find the step that contains the target source path and line number
    currentStepIndex = trace.steps.findIndex((step) => {
      const item = getLast(step.stack);
      return item.path === targetSourcePath && item.line_number === targetLineNumber;
    });
    if (currentStepIndex !== -1) {
      currentStep = trace.steps[currentStepIndex];
      currentStackElement = getLast(currentStep.stack);
    }
  }

  const renderedEnv = currentStepIndex !== null ? renderEnv({trace, currentStepIndex}) : null;
  const renderedLines = renderLines({trace, currentPath, currentLineNumber, currentStepIndex, targetStepIndex, rawMode, animateMode, navigate});

  return (
    <div
      className="trace-viewer-container"
      onMouseDown={handleMouseDown}
    >
      <div className="lines-panel">{renderedLines}</div>
      <div
        className="env-panel"
        style={{
          left: envPosition.x,
          top: envPosition.y,
          cursor: isDragging ? 'grabbing' : 'grab',
        }}
      >
        {renderedEnv}
      </div>
    </div>
  );
}

function stepForward({trace, currentStepIndex, navigate}) {
  const newStepIndex = currentStepIndex + 1;
  if (newStepIndex < trace.steps.length) {
    updateUrlParams({ step: newStepIndex, source: null, line: null }, navigate);
  }
}

function stepBackward({currentStepIndex, navigate}) {
  const newStepIndex = currentStepIndex - 1;
  if (newStepIndex >= 0) {
    updateUrlParams({ step: newStepIndex, source: null, line: null }, navigate);
  }
}

function stepOverForward({trace, currentStepIndex, navigate, stayOnSameLine}) {
  const newStepIndex = getStepOverIndex({trace, currentStepIndex, direction: 1, stayOnSameLine});
  if (newStepIndex < trace.steps.length) {
    updateUrlParams({ step: newStepIndex, source: null, line: null }, navigate);
  }
}

function stepOverBackward({trace, currentStepIndex, navigate}) {
  const newStepIndex = getStepOverIndex({trace, currentStepIndex, direction: -1});
  if (newStepIndex >= 0) {
    updateUrlParams({ step: newStepIndex, source: null, line: null }, navigate);
  }
}

function stepUp({trace, currentStepIndex, navigate}) {
  const newStepIndex = getStepUpIndex({trace, currentStepIndex, direction: 1});
  if (newStepIndex < trace.steps.length) {
    updateUrlParams({ step: newStepIndex, source: null, line: null }, navigate);
  }
}

function getStepOverIndex({trace, currentStepIndex, direction, stayOnSameLine}) {
  // Find the next step that is in the same level (and if not stayOnSameLine, not the same line)
  const currentStep = trace.steps[currentStepIndex];
  let stepIndex = currentStepIndex + direction;
  while (stepIndex >= 0 && stepIndex < trace.steps.length) {
    if (inSameFunction(trace.steps[stepIndex].stack, currentStep.stack) &&
        (!stayOnSameLine || getLast(trace.steps[stepIndex].stack).line_number !== getLast(currentStep.stack).line_number)) {
      return stepIndex;
    }
    if (isStrictAncestorOf(trace.steps[stepIndex].stack, currentStep.stack)) {
      // Escaped
      return stepIndex;
    }
    stepIndex += direction;
  }
  return stepIndex;
}

function getStepUpIndex({trace, currentStepIndex, direction}) {
  // Find the first step that is in the same level
  const currentStep = trace.steps[currentStepIndex];
  let stepIndex = currentStepIndex + direction;
  while (stepIndex >= 0 && stepIndex < trace.steps.length) {
    // If we are no longer in this function but have gone up
    if (!inSameFunction(trace.steps[stepIndex].stack, currentStep.stack) &&
        isStrictAncestorOf(trace.steps[stepIndex].stack, currentStep.stack)) {
      return stepIndex;
    }
    stepIndex += direction;
  }
  return stepIndex;
}

/**
 * Show all the lines up until the current step.
 * Need to also show all the lines that are between the steps.
 */
function computeLinesToShow({trace, currentStepIndex}) {
  const linesToShow = {};
  const pathToLines = {};  // path -> lines

  for (let stepIndex = 0; stepIndex <= currentStepIndex; stepIndex++) {
    const step = trace.steps[stepIndex];
    const stackElement = getLast(step.stack);
    const path = stackElement.path;
    let lineNumber = stackElement.line_number;

    // Also show lines that are before this line up to a line with no indent
    while (true) {
      const loc = getLocation(path, lineNumber);
      if (linesToShow[loc]) {
        break;
      }
      linesToShow[loc] = true;

      let lines = pathToLines[path];
      if (!lines) {
        lines = trace.files[path].split("\n");
        pathToLines[path] = lines;
      }

      const line = lines[lineNumber - 1];
      if (line.match(/^\w/)) {  // No indent
        break;
      }
      lineNumber--;
      if (lineNumber === 0) {
        break;
      }
    }
  }
  return linesToShow;
}

function toggleRawMode({rawMode, navigate}) {
  const newRawMode = !rawMode;
  updateUrlParams({ raw: newRawMode ? "1" : null }, navigate);
}

function toggleAnimateMode({animateMode, navigate}) {
  const newAnimateMode = !animateMode;
  updateUrlParams({ animate: newAnimateMode ? "1" : null }, navigate);
}

function gotoTrace({tracePath, navigate}) {
  const newTracePath = prompt("Enter the path to the trace file", tracePath);
  if (newTracePath) {
    updateUrlParams({ trace: newTracePath, source: null, line: null, step: null }, navigate);
  }
}

/**
 * Represents a location in a file by a string.
 */
function getLocation(path, lineNumber) {
  return `${path}:${lineNumber}`;
}

/**
 * Render the environment variables associated with a step.
 */
function renderEnv({trace, currentStepIndex}) {
  // Go back to previous steps that are in the same frame
  const currentStep = trace.steps[currentStepIndex];

  const envs = [];
  for (let stepIndex = currentStepIndex; stepIndex >= 0; stepIndex--) {
    const step = trace.steps[stepIndex];
    if (inSameFunction(step.stack, currentStep.stack)) {
      if (Object.keys(step.env).length > 0) {
        envs.push(step.env);
      }
    } else if (isStrictAncestorOf(step.stack, currentStep.stack)) {
      // We've gone up to an ancestor
      break;
    }
  }

  envs.reverse();

  // Create the environment by merging all the environments
  const env = {};
  for (const stepEnv of envs) {
    Object.assign(env, stepEnv);
  }

  if (Object.keys(env).length === 0) {
    return null;
  }

  // Create a table mapping variable names to values
  const renderedEnv = Object.entries(env).map(([key, value]) => {
    return (
      <tr key={key}>
        <td className="code-container key">{key}</td>
        <td className="code-container">=</td>
        <td className="code-container">{renderValue(value)}</td>
      </tr>
    );
  });
  return <table className="env"><tbody>{renderedEnv}</tbody></table>;
}

/**
 * Return whether stack1 and stack2 refer to being in the same function (all but
 * the last element must agree).
 */
function inSameFunction(stack1, stack2) {
  if (stack1.length !== stack2.length) {
    return false;
  }
  // Note: don't include the last element in the comparison
  for (let i = 0; i < stack1.length - 1; i++) {
    const a = stack1[i];
    const b = stack2[i];
    if (a.path !== b.path || a.line_number !== b.line_number) {
      return false;
    }
  }
  return true;
}

/**
 * Return whether stack1 is an ancestor of stack2.
 */
function isStrictAncestorOf(stack1, stack2) {
  return stack1.length < stack2.length;
}

function renderValue(value) {
  if (typeof value === "number") {
    if (Math.abs(value) > 1e12) {
      // Use scientific notation
      return value.toExponential(3);
    } if (Math.abs(value) < 1e6) {
      return value.toString();
    } else {
      return value.toLocaleString();  // Put commas in the number
    }
  }
  return JSON.stringify(value, null, 2);
}

function makeProgressBar(currentStepIndex, totalSteps) {
  const progressPercentage = currentStepIndex !== null ? (currentStepIndex / (totalSteps - 1)) * 100 : 0;
  const stepProgress = currentStepIndex !== null ? `${currentStepIndex} / ${totalSteps}` : null;
  return (
    <div title={stepProgress} style={{
      width: '100%',
      height: '4px',
      backgroundColor: 'lightgray',
      marginTop: '4px',
    }}>
      <div style={{
        width: `${progressPercentage}%`,
        height: '100%',
        backgroundColor: '#4CAF50',
        transition: 'width 0.2s ease-out'
      }}/>
    </div>
  );
}

function renderLines({trace, currentPath, currentLineNumber, currentStepIndex, targetStepIndex, rawMode, animateMode, navigate}) {
  const linesToShow = computeLinesToShow({trace, currentStepIndex});

  // Build a map of line number to renderings
  const lineNumberToRenderings = [];
  for (const step of trace.steps) {
    lineNumberToRenderings[getLast(step.stack).line_number] = step.renderings;
  }

  // Get the file contents that we're showing
  const fileContents = trace.files[currentPath];

  // Apply syntax highlighting
  const highlightedContents = hljs.highlight(fileContents, { language: "python" }).value;
  const lines = highlightedContents.trim().split("\n");

  // Render the lines.  For each line:
  // - render the line number
  // - render either the renderings (if they exist) or the line
  const renderedLines = lines.map((line, index) => {
    const lineNumber = index + 1;

    // Replace with renderings if it exists
    const renderings = lineNumberToRenderings[lineNumber];
    const renderedItems = [];
    if (!rawMode && renderings && renderings.length > 0) {
      // Add the indent
      const indent = line.match(/^(\s*)/)[0];
      renderedItems.push(<span key="indent" className="code-container">{indent}</span>);

      // Add all renderings
      const renderedRenderings = renderings.map((rendering, index) => {
        return <span key={index}>
          {renderRendering(rendering, navigate)}
        </span>;
      });
      renderedItems.push(<div key="renderings" className="renderings">{renderedRenderings}</div>);
    } else {
      // Note: line is HTML for syntax highlighting
      renderedItems.push(<span key="code" className="code-container" dangerouslySetInnerHTML={{ __html: line }} />);
    }

    const lineNumberSpan = (
      <span
        key={0}
        className="line-number code-container"
        onClick={() => gotoLine({trace, currentPath, currentLineNumber, currentStepIndex, lineNumber, navigate})}
      >
        {lineNumber}
      </span>
    );

    const renderedItemsSpan = (
      <span>{renderedItems}</span>
    );

    const lineClass = ["line"];
    const isCurrentLine = lineNumber === currentLineNumber;
    if (isCurrentLine) {
      lineClass.push("current-line");
    }
    const location = getLocation(currentPath, lineNumber);
    if (currentStepIndex !== null && animateMode && !linesToShow[location]) {
      lineClass.push("cloaked");
    }

    return (
      <div key={index} className={lineClass.join(" ")} ref={isCurrentLine ? scrollIntoViewIfNeeded : null}>
        {lineNumberSpan}
        {renderedItemsSpan}
      </div>
    );
  });

  const animateIcon = animateMode ? "⛅️" : "☀️";
  const rawIcon = rawMode ? "⚙️" : "⚪️";
  const stepBackwardIcon = "⬅️";
  const stepForwardIcon = "➡️";
  const stepOverBackwardIcon = "↖️";
  const stepOverForwardIcon = "↗️";
  const stepUpIcon = "⤴️";
  const buttons = (
    <span className="icon-buttons">
      <button title="Toggle animation (whether to gradually show content when stepping through) [shortcut: A]" onClick={() => toggleAnimateMode({animateMode, navigate})}>{animateIcon}</button>
      <button title="Toggle raw mode (whether to show the underlying code) [shortcut: R]" onClick={() => toggleRawMode({rawMode, navigate})}>{rawIcon}</button>
      <button title="Step backward (into functions if necessary) [shortcut: h or left]" onClick={() => stepBackward({currentStepIndex, navigate})}>{stepBackwardIcon}</button>
      <button title="Step forward (into functions if necessary) [shortcut: l or right]" onClick={() => stepForward({trace, currentStepIndex, navigate})}>{stepForwardIcon}</button>
      <button title="Step over backward (stay at this level of the stack) [shortcut: k or shift-left]" onClick={() => stepOverBackward({trace, currentStepIndex, navigate})}>{stepOverBackwardIcon}</button>
      <button title="Step over forward (stay at this level of the stack) [shortcut: j or shift-right]" onClick={() => stepOverForward({trace, currentStepIndex, navigate})}>{stepOverForwardIcon}</button>
      <button title="Step forward until we're out of this function [shortcut: u]" onClick={() => stepUp({trace, currentStepIndex, navigate})}>{stepUpIcon}</button>
    </span>
  )

  const header = (
    <div className="header">
      <div className="header-title">
        <span>{currentPath}</span>
        {buttons}
      </div>
      {makeProgressBar(currentStepIndex, trace.steps.length)}
    </div>
  );

  return (
    <div>
      {header}
      <div>
        {renderedLines}
      </div>
    </div>
  );
}

function gotoLine({trace, currentPath, currentLineNumber, currentStepIndex, lineNumber, navigate}) {
  // Find the step that matches the given lineNumber, looking in the direction given by currentLineNumber and lineNumber
  let stepIndex = currentStepIndex;
  if (currentLineNumber <= lineNumber) {
    stepIndex++;
    // Go forward, looking for lineNumber
    while (stepIndex < trace.steps.length) {
      if (getLast(trace.steps[stepIndex].stack).line_number === lineNumber) {
        updateUrlParams({ source: null, line: null, step: stepIndex }, navigate);
        return;
      }
      stepIndex++;
    }
  } else if (currentLineNumber > lineNumber) {
    // Go backward, looking for lineNumber
    while (stepIndex >= 0) {
      if (getLast(trace.steps[stepIndex].stack).line_number === lineNumber) {
        updateUrlParams({ source: null, line: null, step: stepIndex }, navigate);
        return;
      }
      stepIndex--;
    }
  }
  // Otherwise, just show the line
  updateUrlParams({ source: currentPath, line: lineNumber, step: null }, navigate);
}

function scrollIntoViewIfNeeded(el) {
  // Check if element is already in view
  if (!el) {
    return;
  }
  const rect = el.getBoundingClientRect();
  const padding = 50;
  const windowHeight = window.innerHeight || document.documentElement.clientHeight;
  const isInView = rect.top >= padding && rect.bottom <= windowHeight - padding;

  // How much do we have to scroll to get the element into view?
  // If it's not too much, the do smooth scrolling, otherwise do instant scrolling (so it doesn't take too long)
  const scrollDistance = Math.min(Math.abs(rect.top - 0), Math.abs(rect.bottom - windowHeight));
  const behavior = scrollDistance <= 100 ? 'smooth' : 'instant';

  // Only scroll if element is not in view
  if (!isInView) {
    el.scrollIntoView({behavior, block: 'center'});
  }
}

function MarkdownRenderer({ content, style }) {
  const [renderedContent, setRenderedContent] = useState("");

  // Render and set `renderedContent`
  useEffect(() => {
    // Preserve the trailing whitespace
    const trailingWhitespace = content.endsWith(" ") ? "&nbsp;" : "";

    // Because we're rendering only one line of content at a time, the markdown
    // conversion puts <p> tags which produce a lot of vertical space.  So we
    // remove them.
    let markdown = marked(content);
    markdown = markdown.replace(/\n$/, '');
    markdown = markdown.replace(/^<p>/g, '').replace(/<\/p>$/g, '');

    // Add the trailing whitespace back
    markdown = markdown + trailingWhitespace;
    setRenderedContent(markdown);
  }, [content]);  // Only re-run if content changes

  // Trigger MathJax to render
  // TODO: this flickers every time we rerender (step)
  useEffect(() => {
    if (renderedContent && window.MathJax) {
      window.MathJax.typeset();
    }
  }, [renderedContent]);  // If put this, then don't update; otherwise too slow

  return <span className="markdown" style={style} dangerouslySetInnerHTML={{ __html: renderedContent }} />;
}

function ExternalLink({ link, style }) {
  const anchorText = getReferenceAnchorText(link);
  if (!link.title) {
    return <a href={link.url} target="_blank" style={style}>{anchorText}</a>;
  }

  const notes = link.notes && link.notes.split(/\n/).map((line, index) => <div key={index}>{line}</div>);

  const org = link.organization && `[${link.organization}] `;

  return (
    <div className="link-container" style={{ display: 'inline-block', position: 'relative' }}>
      <a
        href={link.url}
        target="_blank"
        style={style}
        className="external-link"
      >
        {anchorText}
      </a>
      <div className="link-hover-panel">
        {link.title && <div className="link-title">{link.title}</div>}
        {link.authors && <div className="link-authors">{org}{renderAuthors(link.authors)}</div>}
        {link.date && <div className="link-date">{renderDate(link.date)}</div>}
        {link.description && <div className="link-description">{link.description}</div>}
        {link.notes && <div className="link-notes">{notes}</div>}
      </div>
    </div>
  );
}

function getReferenceAnchorText(reference) {
  // For citations, use the title
  if (reference.authors) {
    // Get the last name of the first author, add a + if there is more than one author, and add the year
    const lastName = getLast(reference.authors[0].split(" "));
    const plus = reference.authors.length > 1 ? "+" : "";
    const year = reference.date && reference.date.split("-")[0];
    return `[${lastName}${plus} ${year}]`;
  }
  return reference.title || reference.url;
}

function renderDate(date) {
  // Render date as "2025-01-01"
  return date.split('T')[0];
}

function renderAuthors(authors) {
  // authors is a list of strings that could be very long, so take the first 5 and last 5 if needed
  const maxAuthors = 10;
  if (authors.length > maxAuthors) {
    const numOmitted = authors.length - maxAuthors;
    return authors.slice(0, maxAuthors / 2).join(", ") + ` ... (${numOmitted} more) ... ` + authors.slice(-maxAuthors / 2).join(", ");
  } else {
    return authors.join(", ");
  }
}

function renderRendering(rendering, navigate) {
  if (rendering.type === "markdown") {
    return <MarkdownRenderer content={rendering.data.toString()} style={rendering.style} />;
  } else if (rendering.type === "image") {
    return <img src={rendering.data} style={rendering.style} />;
  } else if (rendering.type === "link") {
    if (rendering.internal_link) {
      // Create a link to a particular path, line number
      // TODO: center when we jump to the link
      const link = rendering.internal_link;
      const anchorText = rendering.data || link.path + ":" + link.line_number;
      return (<a href="#" style={rendering.style}
                 onClick={() => updateUrlParams({ source: link.path, line: link.line_number, step: null }, navigate)}
              >
        {anchorText}
      </a>);
    } else if (rendering.external_link) {
      const link = rendering.external_link;
      return <ExternalLink link={link} style={rendering.style} />;
    }
  } else {
    return <span style={rendering.style}>{rendering.data}</span>;
  }
}

function updateUrlParams(params, navigate) {
  const urlParams = new URLSearchParams(window.location.search);
  Object.entries(params).forEach(([key, value]) => {
    if (value === null) {
      urlParams.delete(key);
    } else {
      urlParams.set(key, value);
    }
  });
  navigate(`?${urlParams.toString()}`);
}

export default TraceViewer;