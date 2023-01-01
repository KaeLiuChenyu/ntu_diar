

def join_samespeaker_segments(segments, silence_tolerance=0.5):
        """
        Join up segments that belong to the same speaker, 
        even if there is a duration of silence in between them.

        If the silence is greater than silence_tolerance, does not join up
        """
        new_segments = [segments[0]]

        for seg in segments[1:]:
            if seg['label'] == new_segments[-1]['label']:
                if new_segments[-1]['end'] + silence_tolerance >= seg['start']:
                    new_segments[-1]['end'] = seg['end']
                    new_segments[-1]['end_sample'] = seg['end_sample']
                else:
                    new_segments.append(seg)
            else:
                new_segments.append(seg)
        return new_segments


def rttm_output(segments, recname, outfile=None):
    assert outfile, "Please specify an outfile"
    rttm_line = "SPEAKER {} 0 {} {} <NA> <NA> {} <NA> <NA>\n"
    with open(outfile, 'w') as fp:
        for seg in segments:
            start = seg['start']
            offset = seg['end'] - seg['start']
            label = seg['label']
            line = rttm_line.format(recname, start, offset, label)
            fp.write(line)


def make_output_seconds(cleaned_segments, fs):
        """
        Convert cleaned segments to readable format in seconds
        """
        for seg in cleaned_segments:
            seg['start_sample'] = seg['start']
            seg['end_sample'] = seg['end']
            seg['start'] = seg['start']/fs
            seg['end'] = seg['end']/fs
        return cleaned_segments


def join_segments(cluster_labels, segments, tolerance=5):
    """
    Joins up same speaker segments, resolves overlap conflicts

    Uses the midpoint for overlap conflicts
    tolerance allows for very minimally separated segments to be combined
    (in samples)
    """
    assert len(cluster_labels) == len(segments)

    new_segments = [{'start': segments[0][0],
                     'end': segments[0][1],
                     'label': cluster_labels[0]}]

    for l, seg in zip(cluster_labels[1:], segments[1:]):
        start = seg[0]
        end = seg[1]

        protoseg = {'start': seg[0],
                    'end': seg[1],
                    'label': l}

        if start <= new_segments[-1]['end']:
            # If segments overlap
            if l == new_segments[-1]['label']:
                # If overlapping segment has same label
                new_segments[-1]['end'] = end
            else:
                # If overlapping segment has diff label
                # Resolve by setting new start to midpoint
                # And setting last segment end to midpoint
                overlap = new_segments[-1]['end'] - start
                midpoint = start + overlap//2
                new_segments[-1]['end'] = midpoint
                protoseg['start'] = midpoint
                new_segments.append(protoseg)
        else:
            # If there's no overlap just append
            new_segments.append(protoseg)

    return new_segments


def pre2rttm(cluster_labels, 
        segments, 
        fs, 
        recname, 
        outfile, 
        silence_tolerance = 0.2
        ):
        
  cleaned_segments = join_segments(cluster_labels, segments)
  cleaned_segments = make_output_seconds(cleaned_segments, fs)
  cleaned_segments = join_samespeaker_segments(cleaned_segments,
                          silence_tolerance
                          )
  print('Save prediction')
  rttm_output(cleaned_segments, recname, outfile=outfile)